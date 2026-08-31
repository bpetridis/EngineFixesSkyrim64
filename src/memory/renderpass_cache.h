#pragma once
#include "memory/allocator.h"

#include <algorithm>
#include <bit>
#include <mutex>
#include <new>
#include <vector>

namespace Memory::RenderPassCache
{
    namespace detail
    {
        // Deferred-free quarantine for freed BSRenderPasses.
        //
        // EngineFixes replaces the engine's dedicated BSRenderPass pool with
        // general-allocator alloc/free. The engine's draw path (BSBatchRenderer
        // -> BeginPass -> a_shader->SetupTechnique) can read a BSRenderPass after
        // it was freed: a stale entry left in a pass bucket is iterated, and with
        // immediate free the slot has already been handed to an unrelated
        // allocation, so Pass->shader (offset 0) reads back a garbage vftable ->
        // execute-AV CTD. This is the unfixed root of the Community Shaders conflict
        // (CS issue #1601 -- the CS-side guards only validate Pass->sceneLights[],
        // never Pass->shader). CS background shader compilation removes the
        // render-thread compile stall that previously made the window rare, exposing
        // it on essentially every cell load.
        //
        // CORRECTION (2026-08-31, from the Test 5 result -- see git history): the
        // paragraph above describes why this quarantine was written, and it does
        // guard a genuine hazard, but it is NOT the cause of the MapMenu/
        // ShadowSceneNode CTD this branch chased. A run configured to perform ZERO
        // frees -- nothing aged out, every pass leaked, every lights array leaked --
        // still crashed with the same signature in under four minutes. That falsified
        // every free-timing hypothesis at once. The actual root cause was the
        // sceneLights array size contract; see the comment on SetLights below.
        // The quarantine is kept because it is cheap and the hazard it describes is
        // real, but note that its original justification (matching an observed crash
        // cluster) is now suspect -- that cluster was very likely the sceneLights
        // overread all along. Whether to keep it at all is an open question worth
        // revisiting once the sceneLights fix has been validated in the field.
        //
        // Freed passes are parked intact (including their sceneLights) in a FIFO
        // ring tagged with the engine frame (BSGraphics::State::frameCount) and
        // physically freed only once kQuarantineFrames frames have elapsed -- past
        // any in-flight draw that could still hold a stale reference, so a stale
        // read still sees the original valid shader/lights. The age test is an
        // unsigned subtraction (wrap-safe) and does not assume the counter advances
        // by one per call, so a frozen counter (loading screen, pause) holds passes
        // longer rather than freeing them early.
        //
        // OVERFLOW POLICY. The ring is bounded so memory cannot run away. When it
        // fills, every entry is by construction YOUNGER than kQuarantineFrames --
        // the drain loop in Deallocate has already released everything older -- so
        // freeing the oldest to make room would hand back exactly the memory this
        // quarantine exists to protect, producing the use-after-free it was written
        // to prevent. (That was the pre-7.6.x behaviour and it is reachable in
        // practice: during a fast travel or any load screen the render frame counter
        // stops advancing while the scene graph is torn down and rebuilt, so nothing
        // drains while retires pour in.) The ring therefore DROPS the oldest
        // reference without freeing it -- the pass is leaked, ~72 bytes plus its
        // sceneLights array -- and logs. Leaking a bounded amount is strictly
        // preferable to a UAF CTD, and uRenderPassQuarantineSize lets the ring be
        // sized so overflow does not happen at all. Retire stays allocation-free
        // (the ring is sized once at Install and its capacity is a power of two, so
        // the wrap is a mask, not a division), restoring the safety of the engine's
        // original pool (freed memory stays pass-shaped) while keeping EF's dynamic
        // growth.
        // Frames a retired pass is held before release. Three frames clears any
        // plausible in-flight draw; the test matrix in git history found no evidence
        // that widening it helps (hypothesis H1, frame-count aging vs. a wall-clock
        // hazard, was falsified -- see the sceneLights fix in SetLights for the actual
        // root cause), so this stays a constant rather than a tunable.
        inline constexpr std::uint32_t kQuarantineFrames = 3;
        inline constexpr std::uint32_t kRetiredTag = 0xD1ED0FF5u;  // pad44 sentinel: pass is quarantined

        // uRenderPassQuarantineSize is rounded up to a power of two and clamped to
        // this range. The floor keeps a pathological setting from degenerating into
        // "leak everything"; the ceiling bounds the ring itself at 64 MB.
        inline constexpr std::size_t kMinQuarantined = 1024;
        inline constexpr std::size_t kMaxQuarantined = 4194304;

        // Log the first overflow, then at most one line per this many drops, so a
        // sustained overflow is visible in the log without flooding it. The previous
        // `static bool warned` logged exactly once per session, which understated a
        // sustained overflow by an unbounded factor.
        inline constexpr std::uint64_t kOverflowLogInterval = 4096;


        struct RetiredPass
        {
            RE::BSRenderPass* pass;
            std::uint32_t     frame;
        };
        inline std::vector<RetiredPass> s_ring;
        inline std::size_t              s_mask = 0;   // capacity - 1 (capacity is a power of two)
        inline std::size_t              s_head = 0;   // next write slot
        inline std::size_t              s_count = 0;  // live entries
        inline std::uint64_t            s_dropped = 0;
        inline std::mutex               s_retireMutex;

        inline std::uint32_t CurrentFrame()
        {
            const auto* state = RE::BSGraphics::State::GetSingleton();
            if (state)
                return state->frameCount;

            return 0;
        }

        inline void FreeNow(RE::BSRenderPass* a_renderPass)
        {
            if (a_renderPass->sceneLights != nullptr)
                Allocator::GetAllocator()->DeallocateAligned(a_renderPass->sceneLights);
            Allocator::GetAllocator()->DeallocateAligned(a_renderPass);
        }

        // Drop the oldest quarantined entry. Caller holds s_retireMutex.
        // a_free == false abandons the pass (leak) instead of releasing it; see the
        // overflow policy above for why the overflow path must never free.
        inline void DropOldest(bool a_free)
        {
            auto& slot = s_ring[(s_head + s_ring.size() - s_count) & s_mask];
            if (a_free)
                FreeNow(slot.pass);
            slot.pass = nullptr;
            --s_count;
        }

        // The engine gives every BSRenderPass a FIXED 16-entry sceneLights array.
        // BSRenderPassCache::Init carves the pool's light block into 0x80-byte slices
        // (0x80 / 8 == 16 pointers), one per pass, and stores the slice pointer at
        // BSRenderPass+0x38; the array is never resized and never individually freed.
        // The engine's own SetLights writes numLights entries and then explicitly
        // ZERO-FILLS entries [numLights, 16) -- its `cmp dl, 0x10 / jae` tail. Engine
        // code therefore relies on all 16 slots existing and being initialised, and
        // reads past numLights (shadow lights live in the same array, counted by
        // numShadowLights at +0x20).
        //
        // Sizing this array to numLights, as this patch previously did, turns every
        // such read into a heap overread past the end of a small allocation, handing
        // the engine a garbage BSLight* that it then refcounts. That is a corruption
        // path with no relationship to when anything is freed -- which is why leaking
        // every pass and every lights array (Test 5) did not stop the CTD.
        //
        // Match the engine exactly: always 16 entries, always zero-filled past
        // numLights, never reallocated once attached to a pass.
        inline constexpr std::size_t kSceneLights = 16;

        inline RE::BSLight** AllocateSceneLights()
        {
            auto* lights = static_cast<RE::BSLight**>(
                Allocator::GetAllocator()->AllocateAligned(sizeof(RE::BSLight*) * kSceneLights, 8));
            std::memset(lights, 0, sizeof(RE::BSLight*) * kSceneLights);
            return lights;
        }

        inline void SetLights(RE::BSRenderPass* a_renderPass, uint8_t a_numLights, RE::BSLight** a_lights)
        {
            if (a_renderPass->sceneLights == nullptr)
                a_renderPass->sceneLights = AllocateSceneLights();

            // The engine trusts numLights <= 16 because its array is exactly that big;
            // clamp the copy so a bogus count cannot run off the end of ours either.
            const auto copy = (std::min)(static_cast<std::size_t>(a_numLights), kSceneLights);
            for (std::size_t i = 0; i < copy; ++i)
                a_renderPass->sceneLights[i] = a_lights[i];
            for (std::size_t i = copy; i < kSceneLights; ++i)
                a_renderPass->sceneLights[i] = nullptr;

            a_renderPass->numLights = a_numLights;
        }

        inline void Set(RE::BSRenderPass* a_renderPass, RE::BSShader* a_shader, RE::BSShaderProperty* a_property, RE::BSGeometry* a_geometry, uint32_t a_passEnum, uint8_t a_numLights, RE::BSLight** a_lights)
        {
            a_renderPass->shader = a_shader;
            a_renderPass->shaderProperty = a_property;
            a_renderPass->geometry = a_geometry;
            a_renderPass->passEnum = a_passEnum;
            a_renderPass->accumulationHint = 0;
            SetLights(a_renderPass, a_numLights, a_lights);
        }

        inline RE::BSRenderPass* Allocate(RE::BSShader* a_shader, RE::BSShaderProperty* a_property, RE::BSGeometry* a_geometry, uint32_t a_passEnum, uint8_t a_numLights, RE::BSLight** a_lights)
        {
            constexpr std::size_t size = sizeof(RE::BSRenderPass);
            auto*                 data = Allocator::GetAllocator()->AllocateAligned(size, 8);
            memset(data, 0, size);

            auto* renderPass = static_cast<RE::BSRenderPass*>(data);

            renderPass->shader = a_shader;
            renderPass->shaderProperty = a_property;
            renderPass->geometry = a_geometry;
            renderPass->passEnum = a_passEnum;
            renderPass->accumulationHint = 0;
            renderPass->extraParam = 0;
            renderPass->LODMode.index = 3;
            renderPass->LODMode.singleLevel = false;
            renderPass->numShadowLights = 0;
            renderPass->next = nullptr;
            renderPass->passGroupNext = nullptr;
            renderPass->cachePoolId = 0xFEFEDEAD;
            renderPass->pad44 = 0;

            SetLights(renderPass, a_numLights, a_lights);

            return renderPass;
        }

        inline void Deallocate(RE::BSRenderPass* a_renderPass)
        {
            // Do NOT touch a_renderPass's payload here: a late/concurrent draw may
            // still dereference it. Park it intact; it is physically freed only once
            // kQuarantineFrames frames have elapsed. See the quarantine note above.
            std::scoped_lock lock(s_retireMutex);

            // Skip a double Deallocate of the same pass (would double-free on drain).
            // Allocate stamps pad44 as 0; we restamp it on retire below.
            if (a_renderPass->pad44 == kRetiredTag)
                return;

            // Sample the frame under the lock: a pre-lock read could be backdated by
            // contention, under-quarantining this pass (drained before kQuarantineFrames).
            const auto now = CurrentFrame();

            // Drain everything old enough to be past any in-flight reference.
            while (s_count > 0) {
                const auto& oldest = s_ring[(s_head + s_ring.size() - s_count) & s_mask];
                const auto  age = now - oldest.frame;
                if (age < kQuarantineFrames)
                    break;
                DropOldest(true);
            }

            // Safety valve: never exceed the ring (extreme churn / frozen counter).
            // Everything still queued is younger than kQuarantineFrames, so the
            // oldest is abandoned rather than freed -- see the overflow policy above.
            if (s_count == s_ring.size()) {
                if (s_dropped++ % kOverflowLogInterval == 0) {
                    logger::warn(
                        "render pass quarantine full ({}); leaking oldest pass rather than freeing it early "
                        "({} leaked so far) -- raise uRenderPassQuarantineSize"sv,
                        s_ring.size(), s_dropped);
                }
                DropOldest(false);
            }

            a_renderPass->pad44 = kRetiredTag;
            s_ring[s_head] = { a_renderPass, now };
            s_head = (s_head + 1) & s_mask;
            ++s_count;
        }

        // Returns the ring capacity to use: uRenderPassQuarantineSize clamped to
        // [kMinQuarantined, kMaxQuarantined] and rounded up to a power of two so the
        // wrap in Deallocate stays a mask.
        inline std::size_t ResolveCapacity()
        {
            auto requested = static_cast<std::size_t>(Settings::MemoryManager::uRenderPassQuarantineSize.GetValue());
            requested = std::clamp(requested, kMinQuarantined, kMaxQuarantined);
            return std::bit_ceil(requested);
        }

        inline bool Install()
        {
            const auto capacity = ResolveCapacity();
            try {
                s_ring.assign(capacity, RetiredPass{ nullptr, 0 });
            } catch (const std::bad_alloc&) {
                logger::error("render pass cache: could not allocate a {}-entry quarantine, patch NOT installed"sv, capacity);
                return false;
            }
            s_mask = capacity - 1;
            s_head = 0;
            s_count = 0;
            s_dropped = 0;


            REL::Relocation allocate{ RELOCATION_ID(100717, 107497) };
            REL::Relocation deallocate{ RELOCATION_ID(100718, 107498) };
            REL::Relocation setlights{ RELOCATION_ID(100711, 107490) };
            REL::Relocation init{ RELOCATION_ID(100720, 107500) };
            REL::Relocation kill{ RELOCATION_ID(100721, 107501) };

            allocate.replace_func(VAR_NUM(0x9A, 0xF9), Allocate);
            deallocate.replace_func(VAR_NUM(0x60, 0x68), Deallocate);
            setlights.replace_func(0x69, SetLights);
            if (!REL::Module::IsAE()) {
                REL::Relocation set{ REL::ID(100710) };
                set.replace_func(0x90, Set);
            }

            init.write_fill(REL::INT3, VAR_NUM(0x1BD, 0x1BF));
            init.write(REL::RET);
            kill.write_fill(REL::INT3, 0xAB);
            kill.write(REL::RET);
            REL::Relocation clear{ RELOCATION_ID(100722, 107502) };
            clear.write_fill(REL::INT3, VAR_NUM(0xE7, 0x16B, 0xB7));
            clear.write(REL::RET);
            return true;
        }
    }

    inline void Install()
    {
        if (detail::Install())
            logger::info("installed render pass cache patch (quarantine {} entries)"sv, detail::s_ring.size());
    }
}
