# Historical continuation before final delivery

Archived on 2026-09-11. This is historical execution context; current status is in [CONTINUATION.md](CONTINUATION.md). Training, measurement, and once-only test work mentioned as pending below subsequently completed. Do not use these old commands or session ownership as current instructions.

# Current execution checkpoint

Continue the active approved goal through the final paired revision, inference profiling/cache experiment, frozen final test once, selected prediction-only replay QA/recording and final handoff. BaselineA/B, revision1 and revision2 are complete. Third and final learning revision passed recovery and both production gates; paired training is running (control first). Do not stop at trained checkpoints.

## Current ownership

- GPU: root paired launcher exec83265, artifacts/revision03/run_pair.py, recoveryreport artifacts/revision03-recovery/attempt01/report.json passed. It runs control/balanced production50update gates then full8epochs each sequentially. Live execution state artifacts/revision03/execution.json. Dataset worker releasedGPU. Do not start another GPU job.
- Queued endpoint evaluation exec25214 artifacts/revision03/evaluate_pair.py waits for paired execution passed, then ownsGPU for control/balanced/parent normalval, endpointproposals andfixed96trainingfit. No source/config/decision/helper changes during the queue.
- Model worker finished the focal normalization source/tests and is reviewing root artifact launchers CPU-only. Metrics worker is reconstructing original A/R2 common-cohort results and reviewing paired protocol CPU-only.
- Root owns artifacts/revision03/production_gate.py and run_pair.py, configs/decision/docs. Production source frozen at02d7eb2366a6c833caca6687622c9614b8ad97b43885b17c0b8e172d49b010f5; do not edit during gate/training/evaluation.
- BaselineA validation viewer port8770/session70681 is provisional, not final selected replay.

## Final paired revision

Decision artifacts/revision03/decision.json; configs/revision03-control.yaml and revision03-balanced.yaml. Same R2best parent SHA3dfa52ea6a14d3f080a7d8507489c550419425969e8006773a91502706f9aaf0; full model trainable, original AdamW1e-4/seed42/AMP1024/accum4, eight fixed epochs, cosine8, matching IoUweight2. Only positive center loss reduction differs. Global remains default; present-class candidate leaves negative term/masks unchanged. All8diagnostic class sums/counts detached and excluded total.30focused CPUcontrols/78combined passed,2CUDAexcluded. Epoch8 endpoints prespecified; no lucky epoch selection. Control gains mean extra training, not loss success. Candidate-specific pedestrian/mechanism/geometry guards in decision; no test tuning. Three revisions now consumed, no fourth.

Root paired launcher command once recovery report passes: `.venv/Scripts/python.exe artifacts/revision03/run_pair.py --recovery-report EXACT_REPORT_PATH`. It gates both50update starts, then resumes full8epochs sequentially. Source/config/parent/decision bound. Keep successful checkpoints, logs, optimizer/scaler/RNG/history/progress. Full-network grid_sample backward is known nondeterministic; recovery must verify exact restoration and report independently measured next-update variability, not claim bitwise all-training replay.

## Completed models

A24epochs best21: road0.8157808311, sidewalk0.4988688348, carAP0.3742230969, pedestrianAP0; fixed composite0.42221819. A SHA0c2c8ffc34f951669011b9f00041e2606f627384d64a5f1354367f2ff38326d2. B9epochs best3 fixed composite0.406504; R1nineepochs best1 composite0.420788, not promoted.

R2nineepochs9675updateszeroAMPskips, exact frozen base. Bestepoch1 ordinary road0.8157808311, sidewalk0.4988866686, carAP0.3775182458, pedestrianAP0; composite0.4230464364. LatestendpointcarAP0.3010003575. Completion report/6plots/doc/ledger updated. Both train/depth/road fit improve over heldout; fixed96trainingfit bestcarAP0.800085 and pedAP0.200391 (17/59). No validation pedestrian matches:6observations1identity, all6lack sameclassproposalwithin2m. R2 source-era602e frozen reports remain valid historical evidence.

R2ordinary repeat artifact cacheOFF under source602e completed: artifacts/profiling/lift-cache-prototype/ordinary-off.json, carAP0.3775381741. Exactsmall A/R2 score difference exceeds observed repeat numeric change, not a significance claim. Common-cohort work pending. Final current-source ordinary validation still needed after last revision.

## Cache/profile evidence

Artifact lift cache prototype byte-exact406fields/14actualR2frames strictGPUparity, source602e; cold/warm, fullhistory, calibration/dropout allpassed,9hits5builds,7.40MiBcache. It is not production-integrated or accepted. Source changed only loss since that receipt; final cache parity must bind current source/selected weights. Original wrapper artifacts/profiling/lift-cache-prototype/run_candidate.py supports cacheoff/on ordinaryval and benchmark; keep runtime_extension identities alongside canonical source. No final50warmup500timedx3 benchmark yet. R1preliminary30hookprofile modelmedian19.678ms; actualcompile probe failed TritonMissing. No compiled or30FPSclaim. Full optimization guard has missing pedestrian geometry => not_verifiable, never automaticpass.

## Data/quality/final obligations

Manifest804f711286751ef5ed3bea1f04a9969c8f58d34cfc8ab05490e6b57daa9d9a0c;4300train197val1622sealedtest. Previous environment/data/geometry/evaluation/pipeline/learnability gates passed. Four native cameras/radialdepths, flat TRAIN-derivedground-0.930941852176808m. Five-native-frame sampling measured1.913Hz, actualtimestamps used. Sparsegroundvalidity and unsupportedregionsmasked; some validation geographic overlap. Sparse scored boxes leave~98.9%of displayed A/Bboxes unassessed. Visuallyconfirmed offroadclutter; no groundtruthfilter or output smoothing accepted. Confidencefilter loses real matches aswell; finaldisplaythreshold undecided.

After paired endpoints/diagnostics: select with fixedmean road/sidewalk/carAP/pedAP and predeclared experiment guards; finalize source, revalidate, profile50warmup500x3 and comparecacheaccuracy/speed. Freeze exactweights/config/source/manifest; evaluate sealedtest once logicalrun withreceipt, no test tuning. Export selectedpredictions, create metadata including predictions_sha256 and checkpointSHA, fill artifact replay QAplan from datasetworker, run CPUChromiumcontrols/screenshots/WebM, inspectactualimagesandvideo. Report CPUrender scopes separately fromfreshGPUinference. Deliver checkpoint,config/lock,manifest,experiments/plots,finaltest/latency,recordings andonecommand instructions, allunmettargets. Only then markgoalcomplete.

Earlier detailed state: CONTINUATION_HISTORY_THROUGH_REVISION1.md and run/evaluation artifacts.

## R3 production start evidence

Both50update gates passed all8intended group gradients andweightchanges, exactsameparentinitialization, zeroAMPskips, history3, epoch0nextstep200, schedulerhorizon8. Firstgate work control20.652917600004s, balanced recorded in its production-gate.json. Add each gate work to resumed firstepoch time in final accounting. Recovery reportSHA215885ff55807dc5f83ad021898fde2470787d214b97f7343e8a90b04e131d54; large orderedhistory differences within repeat envelope are notsmoothnessproof. Gate report paths artifacts/revision03/{control,balanced}/production-gate.json. LauncherSHA3a0e7498aeb2b38c1957184b080659b22c8cf33b55e2bf10d8fbad905d7811cb, productionhelperSHA23c723fa153bcb7034cf3b8468f044b32a9bb1fc4f190a23e4a46638065baf78, decisionSHA2d5213516604ee1a40e1b0fa5c792f131ae6d3af012c6a1bfd17eca7b5471d91. Do not edit these while running.

A/R2common-cohort finished exactoriginalmetricsreproduction, artifacts/final-handoff-planning/common-cohort/a-vs-r2/report.json.111sharedcars37IDs62transitions: heading39.129->39.267deg; temporalheading21.322->21.229deg. OriginalA source7332 not602e; both preservedtransparent. Metricsworker now prepares CPU R3 endpoint assessor, authorizedvaltargets only, noGPU.

## Live update after first control epoch

Controlepoch1 completedscore0.3717023492, road0.7828377757, sidewalk0.4238974451, carAP0.2800741762, pedAP0; zeroAMPskips. Completefirstepochwork328.1873243+gate20.6529176=348.8402419s. Initialremainingestimate86.9minfor15epochs (minusbalancedalreadyperformedgate), artifacts/revision03/completion-estimate.json. Preservefixed8endpoint,noadaptivechangesafterearlyweakcontrol. Bothproductiongatespassedandrootlauncher83265runningcontrol; evaluatequeue25214waiting.

Metricsworker completed55CPUcontrol endpointassessor: `.venv/Scripts/python.exe artifacts/final-handoff-planning/revision03-assessor/assess_pair.py assess --allow-validation-targets --output artifacts/revision03-assessment` ONLYafterpairedtraining+7evaljobspass. ItreadsvaltargetsonlyandreproducesoriginalmetricsthencomparescommoncarfieldsandpedAP/mechanismguards. `assessment.json` statuspass/reject/inconclusive, validrejectionexit0. Earliermodelbestsoverviewnotfinalfreeze.

ModelworkerCPUprobesNMSsaved96realStageAinitialproposalarrays: originalmedian12.40/22.29/22.09ms across3repswithconcurrentloadvariation. MaterialCPUcost; artifact-only exactdirectedclass-wiseprecomputedIoUmatrix prototypepending under artifacts/profiling/nms-prototype. Sourceunchanged; noGPU work. FinalselectedGPU/ordinaryparity/timingstillrequired. Datasetworkerpreparing handoffmanifestbuilderdisableduntilfinalledgercomplete and assistantvisualreviewhashreceipt. No humanapproval requirement.

## Inference candidate and finalartifact preparation

NMSv1 fullmatrix no repeatablegain, retainednegativeevidence. NMSv2 cachedper-boxgeometry preservesactualgreedypairs, all96savedrealoutputsbyteexact/20CPUcontrols; threepairedCPUrepsmedian22.036->6.214ms, p9524.695->7.870ms, sourceunchanged. ScopeCPUStageAtrainingproposalsunderconcurrentload, notfinalFPS. Module artifacts/profiling/nms-prototype/cached_geometry_nms.py SHA6b72150b28eb0dad4ccff13685e34f3ce71318c4fe8ddc0fb7469193993c5fa8, export select_detections_cached_geometry same5primaryargs. Modelworkerpreparesactualval197same-outputdecoder+separatetrackerparityrunner, GPUexecutiondisableduntilR3training/evalpasses.

Root unified artifact wrapper artifacts/profiling/run_backend_candidate.py prepared/py_compile/helpchecked, NOTexecuted. CLI evaluate|benchmark --lift-cache off|on --nms original|cached_geometry [--lift-parityreceipt] [--nms-parityreceipt] thenstandard --checkpoint --splitval --output...; benchmarkrequires--threshold0.01. Rejectstest/FP32/compile/diagnostics. MonkeypatchesdecoderBEFOREengineimport, bindsallprototypes/wrapper/runtimechoiceintoreport+sidecar, requiresactualselected-modelparityreceipts; canonicalsourceunchanged. Cancompareoriginal, NMSonly, cacheonly, combinedincrementally; eachfullval/timingmustmatchactualextensioncontract. Strictbackendmissingpedmetricsremainnot_verifiable, noautomaticpromotion. Oldlift-onlywrapperreportsoffsource602e retainedhistorical.

Datasetworkerprepared build_handoff_manifest.py + handoff-plan.template.json + HANDOFF_BUILDER.md under artifacts/final-handoff-planning;10syntheticCPUcontrolspass. Builder refusesreadsbeyondincompletefinalledger andrequiresASSISTANTvisualreviewreceiptwithQA/screenshots/video(orviewedvideoframes)hashes. Nohumanapprovalrequired. Allowsunmetaccuracy/FPSandexplicitrejected/inconclusiveoptimizationoutcomes. Actualfinalmanifestgenerationpending.

ControlisolatedAMPskip at epoch4step3424 scale4096->2048 recoveredfinite; recordactualattempts/successfulupdates, donotclaimequalupdatesifarms differ. Metricsworker auditingassessorforproperupdates+skips8600handling,thenpreparingselectedvalconfidence-displayselector (grid.30..95 step.05, highestretaining90%ofeachclassoriginalTPwhereavailable; absentpedunavailable, fallback.30). Presentationonly, noAP/modelselection/cutoff.01changes, onlyvaltargets, no testtuning. Defaultnotyetchanged.


### Control endpoint complete; balanced arm running

Revision3 control training passed all eight epochs with 8,597 accepted updates and three isolated AMP backoffs (epoch4 step3424; epoch6 steps2852 and4088). Epoch8 endpoint composite0.4179383034, car AP0.3502698276, pedestrianAP0, roadIoU0.8187737952, sidewalkIoU0.5027095906. Best epoch6 remains diagnostic only under the frozen fixed-endpoint design. The same paired launcher session83265 now runs balanced training; queued evaluator25214 remains GPU-gated until both arms finish. Source remains02d7eb2366a6c833caca6687622c9614b8ad97b43885b17c0b8e172d49b010f5. Artifact-only optimization wrapper guard fixes do not alter training source.
