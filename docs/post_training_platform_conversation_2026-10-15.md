# Post-training platform MVP — conversation

Date: 15 October 2026

This transcript contains the user messages and substantive assistant replies in this conversation, from the morning greeting through the request to export. Tool calls, internal instructions, and brief tool-progress updates are excluded. Citation widgets are represented by source links collected at the end. The discussion is preserved as it developed, including proposals later revised or paused; it is not a corrected technical specification.

---

## 1. User

good morning. today is going to be a great day. we will make so much progress with MVP, in fact we can finish the first version.

## Assistant

Good morning, Gonca 💙 Yes—today we can turn all that learning into a working first version.

**Today’s finish line: submit a training job → execute it on the GPU → track its status → save the adapter → test it with inference.** One complete path working end to end. That’s a real MVP.

Where did you leave the implementation yesterday?

---

## 2. User

we did one super small SFT training with Ray TRL . The thing is we do not have the data to do something meaningful in terms of the metrics we want to observe.

## Assistant

That tiny SFT run is already a useful milestone: **Ray and TRL can execute training.** Now we need a dataset and evaluation task that make the results interpretable.

**Which metrics do you mean—model quality, platform performance, or both?** They need different things:

| What we want to observe | What we need |
|---|---|
| Training loss and validation loss | Training examples plus a separate validation split |
| Task improvement after SFT | A clearly defined task and held-out test examples; compare the base model with the adapter |
| Tokens/sec, GPU memory, step time | Enough training steps and representative sequence lengths |
| Queue time, job reliability, tenant isolation | Multiple submitted jobs and controlled failure scenarios |

For this first version, **a narrow task with an automatically checkable answer** would give us the clearest signal—for example, converting a customer request into a fixed JSON schema. We could measure valid JSON, schema compliance, and correct field values before and after training. Synthetic data could work, provided the test examples are independently constructed and include different wording.

We don’t need a huge dataset to make the MVP meaningful. We need a task the base model doesn’t already solve perfectly, a clean test set, and enough work to exercise the platform.

**Which base model did you train, and what metrics had you planned to track?**

---

## 3. User

We did one SFT LoRA training as they introduced. Sorry, we did one RAG post training using Hugging Face TTS with gRPC, as introduced in the link I'm going to send you. It's an array document. It's called RAG post training using Hugging Face TTS gRPC. So we can do that, right? And also we can do one LoRA SFT using our own code, but it's super small. Our dataset is like four points or so. Anyway, so what I need is the missing piece for me for this MVP is what would be a good example, multi-tenant example where I can highlight why it matters, how we submit the jobs in terms of at least GPU usage, SAM usage, and job completion time, throughput, all these metrics that matter, that tells us about, that highlights something about choices we make for multi-tenant cases matter. So we need to design one multi-tenant post training use case. We just want to be able to highlight our design decisions in different layers of the post training platform matter. You know what I mean?

## Assistant

**Yes—now I understand. You want a multi-tenant workload that makes our platform design decisions measurable.** The dataset needs to generate realistic training work so we can show how scheduling, resource allocation, isolation, and recovery affect different customers.

Send me the Ray document when ready; some of the tool names seem to have been mixed up in transcription.

I would use **three companies fine-tuning separate LoRA adapters on the same supported base model**, each with private customer-support data:

| Tenant | Workload | What they care about |
|---|---|---|
| **A: Large retailer** | Large training job with long support conversations | Training throughput and finishing within its deadline |
| **B: Small software company** | Small adapter update with short support requests | Quick turnaround, even while A has work queued |
| **C: Travel company** | Medium job with longer requests; submit several jobs | Predictable progress and reliable completion |

These can use subsets of one dataset for the demonstration. Each tenant gets separate dataset ownership, job records, and adapter outputs. **Sharing the base-model ID does not automatically mean training workers share its GPU-memory allocation.**

The most revealing experiment is **running the same submitted workload under different platform policies**:

| Design decision | Experiment | What it demonstrates |
|---|---|---|
| **Scheduling and fairness** | Queue several A jobs, then B’s small job. Compare FIFO with tenant-aware scheduling. | Whether one customer’s backlog makes another customer wait excessively |
| **GPU admission control** | Compare one training job at a time with two concurrent jobs, if memory permits. | Whether concurrency improves total throughput or causes contention, slower jobs, or OOM failures |
| **Training configuration** | Change microbatch size or sequence length for a controlled run. | How configuration affects peak GPU memory, tokens/sec, and job duration |
| **Checkpoint and retry policy** | Interrupt a running job and resume it from a checkpoint. | How much completed work survives, how long recovery takes, and the cost of checkpointing |
| **Tenant isolation** | Have B request A’s dataset or adapter. | Whether the API and artifact-access layer enforce ownership |

For scheduling, an important detail: **if A is already running and occupies the only GPU, tenant-aware queue ordering alone cannot start B immediately.** It can select B next; interrupting A requires explicit preemption and checkpoint/resume support. That gives you a concrete trade-off to explain.

For each job, record:

- **Queue time:** submission → execution starts.
- **Execution time:** execution starts → completion, with model loading and training time broken out.
- **Total completion latency:** submission → completion.
- **Training throughput:** non-padding training tokens processed per second, using a consistent measurement window.
- **Peak GPU memory and host RAM**, plus GPU utilization over time.
- **Failures, retries, and recovery time.**

Across the whole experiment, measure **total elapsed time to finish all jobs, aggregate training throughput, and per-tenant waiting times**. A policy can improve overall throughput while making a small customer wait much longer—that is precisely the behavior we want to expose.

**For today’s first version, I’d implement the three-tenant workload, sequential GPU execution, and FIFO versus tenant-aware queue selection.** Use enough examples for jobs to run for several minutes, with deliberately different job sizes. Then add the concurrency comparison and failure recovery as separate experiments.

This gives the MVP a clear question: **“How do we keep shared GPU capacity productive while giving different customers fair, reliable training service?”**

---

## 4. User

[https://docs.ray.io/en/latest/train/examples/transformers/transformer_reinforcement_learning/README.html#rl-post-training-using-hugging-face-trl-with-grpo](https://docs.ray.io/en/latest/train/examples/transformers/transformer_reinforcement_learning/README.html#rl-post-training-using-hugging-face-trl-with-grpo)

## Assistant

Yes—this clarifies it: **Ray Train + Hugging Face TRL + GRPO**. The example runs reinforcement-learning post-training on math problems, using Qwen2.5-0.5B and 100 examples from DeepMath-103K. It generates multiple answers per prompt and rewards mathematical correctness. The shown code does **not** configure LoRA.

**We can use this alongside your SFT-LoRA code to create a more interesting multi-tenant demonstration:**

| Tenant | Submitted work | Why it matters |
|---|---|---|
| **A: Math tutoring company** | A GRPO job using the Ray example | Includes answer generation, reward computation, and training |
| **B: Customer-support company** | A small SFT-LoRA update | Needs quick turnaround while other tenants have larger jobs |
| **C: Customer-support company** | Several larger SFT-LoRA jobs | Creates a backlog that tests fairness and admission control |

These are proposed workloads; we should **measure their actual memory use and duration** before assigning resource budgets.

The central experiment becomes: **What happens to B’s small job when A and C are using the platform?**

1. **FIFO baseline:** B waits behind previously submitted jobs.
2. **Tenant-aware scheduling:** B gets a turn ahead of C’s remaining backlog, once the running job finishes.
3. **Optional concurrent execution:** If measured memory requirements permit, test whether running two jobs together improves completion times or creates harmful contention.

Keep the submitted jobs and training configurations identical between scheduling experiments. Compare **B’s waiting time, each job’s completion latency, total workload completion time, GPU utilization, peak VRAM, and host RAM**. Report SFT training throughput and GRPO generation/training throughput separately—their tokens represent different work.

**That is the missing piece: a reproducible workload with competing customer needs.** We can keep your existing training implementations and build the experiment around them.

Are you currently using **one GPU, and which model/how much VRAM does it have?**

---

## 5. User

Okay, but that's the problem. How am I going to find this customer support company data? I don't have that. I can only use open source data, right? So maybe you go and look for datasets that I can use. By the way, they can be visual language model datasets that require more memory. I don't want something that require more compute. It can be these different things. Like help me with finding those things, designing those things.

## Assistant

**Yes—you’re right. I gave you hypothetical companies when you needed actual datasets and an executable experiment.** We can simulate tenants entirely with public data; you don’t need access to any company’s private records.

I found two useful datasets that let us build three distinct workloads without making GRPO’s generation cost the centerpiece.

**1. Bitext customer support: your lightweight text workload**

[Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

It contains about **26,900 examples**, with `instruction`, `response`, `intent`, and `category` fields, covering 27 intents. Its listed license is **CDLA-Sharing-1.0**. It is synthetic customer-support data.

We can use it in two ways:

- **Intent routing:** customer request → intent label. This gives us short sequences and a straightforward accuracy metric.
- **Response generation:** customer request → supplied support response. This produces longer training sequences.

For your first tenant, I’d choose **intent routing**. It makes evaluation simple: did the model generate the correct label?

**2. ChartQA: your visual workload**

[ChartQA dataset](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)

It provides **chart images, questions, and answer labels**, with 28,299 training examples and separate validation/test splits. The repository lists **GPL-3.0**. We only need a small subset.

The task is concrete: **image of a chart + question → answer**. It gives us an image-loading pipeline, vision preprocessing, and VLM training to exercise.

**My proposed experiment: three tenants, two datasets**

The following sizes are starting configurations, not measured resource estimates:

| Tenant | Actual dataset/task | Initial job configuration | Platform behavior to expose |
|---|---|---|---|
| **A: Support routing** | Bitext instruction → intent | 512 examples; text LoRA; short sequences | Small jobs needing quick turnaround |
| **B: Chart assistant** | ChartQA image + question → answer | 128 examples; VLM LoRA; lower image-token budget | Multimodal preprocessing and moderate resource demand |
| **C: Detailed-chart assistant** | A separate ChartQA subset, same task | 128 examples; same VLM and LoRA setup; higher image-token budget | Larger per-job memory demand and admission decisions |

For B and C, use the **same VLM** so model size does not confound the comparison. Separately, run the **same ChartQA samples at both image settings** to isolate the effect of image-token budget.

**More memory does not have to mean a long training run.** Higher image-token counts can increase both memory and computation per step; we cannot promise a memory-only increase. But we can keep total compute bounded with small subsets and a fixed, short step budget. Also, the processor must actually produce different image-token counts—changing the source image dimensions alone may not accomplish that.

**How we make the design choices visible**

First, profile each workload alone for approximately **10–20 training steps**, including warm-up. Measure peak VRAM, host RAM, step time, and loading time. These measurements determine whether any pair can safely coexist.

Then replay this submission pattern:

> C submits three jobs. While C1 runs, A submits one small job and B submits one visual job.

Compare:

| Policy | What we learn |
|---|---|
| **FIFO, one job at a time** | Baseline waiting times and total completion time |
| **Tenant round-robin, one job at a time** | Whether A and B get service sooner instead of waiting behind C’s entire backlog |
| **Memory-aware concurrency, if profiling permits** | Whether overlapping compatible jobs improves completion times enough to justify contention |

Round-robin here means choosing among tenants **when a job finishes**; it does not interrupt C1. It also provides turns, not equal GPU-time shares.

This maps directly to your layers:

- **API/data layer:** validate dataset ownership and text-versus-image compatibility.
- **Scheduler:** choose the next tenant and admit only resource-compatible jobs.
- **Training worker:** apply the approved batch/image settings and report measurements.
- **Artifact layer:** save each tenant’s adapter and configuration separately.

**I would start with Bitext intent SFT plus ChartQA VLM SFT.** Keep your existing GRPO example as an additional workload later. This gives you real data, measurable task outputs, and contrasting resource profiles—with a deliberately small training budget.

---

## 6. User

Can we find any other VLM example with public dataset that's already used in VLM post training, something in the autonomous driving area?

## Assistant

**Yes—and DriveLM looks like a much better fit for your MVP and your autonomous-driving background.** I found several datasets that have actually been used for VLM post-training.

| Dataset / example | What the VLM learns | Practical fit |
|---|---|---|
| **[DriveLM](https://github.com/OpenDriveLab/DriveLM)** | Driving-scene questions involving perception, prediction, and planning | **My first choice:** official baseline training instructions, sample annotations, and sample images |
| **[LingoQA — Wayve](https://github.com/wayveai/LingoQA)** | Answering questions about driving videos | Strong option for temporal workloads; the authors used it to fine-tune their VLM baseline |
| **[BDD-X / Dolphins](https://vlm-driver.github.io/)** | Describing and explaining driving behavior from video | Published driving-specific instruction tuning using BDD-X |
| **[DriveLMM-o1](https://github.com/ayesha-ishaq/DriveLMM-o1)** | Driving questions with step-by-step explanations | Relevant, but requires separate nuScenes assets; its dataset card currently lists the license as unknown |

These are documented training applications, rather than datasets I’m merely guessing could work.

**The particularly useful discovery: DriveLM includes a small runnable data example.**

Its official [challenge walkthrough](https://github.com/OpenDriveLab/DriveLM/blob/main/challenge/README.md) supplies:

- `train_sample.json` and sampled images directly in the repository.
- Instructions for training and evaluating its baseline.
- Links to a larger DriveLM image subset and QA annotations, so starting does not require downloading all of nuScenes.

We can use the sample to validate our loader, then select **a few hundred QA examples** for meaningful platform measurements. Adapting these to our chosen VLM and TRL will still require a dataset conversion and image-aware collator; the official baseline is not a drop-in replacement for your text trainer.

**Here is the driving workload I would design.**

Use one small VLM with separate LoRA adapters for simulated customers:

| Tenant | Workload | Controlled difference |
|---|---|---|
| **A: Quick scene-understanding update** | Small DriveLM subset | Few training steps, lower visual-token budget |
| **B: Detailed scene-understanding update** | Comparable DriveLM examples | Higher visual-token budget, same model |
| **C: Fleet adaptation batch** | Larger DriveLM subset, several submitted jobs | More total work and a queue backlog |

Preserve every camera view needed to answer a question. We should **not remove views arbitrarily** just to make a workload smaller. For a controlled memory comparison, process the same examples at two supported image resolutions/token budgets.

This lets you investigate three separate questions:

1. **Fairness:** Does A wait behind all of C’s jobs?
2. **Admission:** Can A and B coexist within measured VRAM limits?
3. **Configuration:** How much do image-token budgets change peak memory, step time, and answer quality?

Keep evaluation scenes separate from training scenes—different questions about the same frame should not land on opposite sides of the split.

**My recommendation is DriveLM first, with LingoQA as the next extension if you want temporal/video inputs.** That gives you a credible driving-domain post-training example while keeping the first implementation focused. Public access does not imply unrestricted licensing: DriveLM’s code license and the underlying image-data terms need to be treated separately.

---

## 7. User

Okay, so DriveAlign seems relevant. I just want to understand this DriveAlign is a dataset, and it can be used to benchmark your vision language models. Is that right?

## Assistant

Do you mean **DriveLM**, the one we just discussed? If so, **yes—it provides a dataset that can be used both to fine-tune and to benchmark vision-language models on driving tasks.**

Each example pairs **driving-scene images and a question with a reference answer**, covering perception, prediction, or planning.

- **Fine-tuning:** use the training examples to update the model—for instance, through LoRA SFT.
- **Benchmarking:** give the model held-out images and questions, then evaluate its answers against the references using defined metrics.

For our MVP, we could **evaluate the base VLM → fine-tune on a small training subset → evaluate the adapted VLM on the same held-out set**, while measuring memory, throughput, and completion time.

That measures driving-scene understanding; it does not by itself establish real-world driving ability. If you meant a separate project called **DriveAlign**, send its link so I can distinguish them.

---

## 8. User

Great. This also gives me an opportunity to do my own narration, because in Bosch I implemented this platform mostly for groups that wanted to do autonomous driving applications, right? Where they think about putting these models on cameras, embedding them on cameras, or using these models in their own autonomous driving solution pipelines and stuff like that. So this was the motivation that I had to come with this post-training platform at Bosch. Now I want to, of course, I cannot use their own datasets anymore. So I can still highlight how I did it using public datasets. DriveLM is one of them. Maybe there are other datasets out there that I can add. But DriveLM is maybe good enough for the MVP. So yeah, let's also work on how I narrated, how I did this platform, implemented such a platform in the past, like which parts of this platform were highlighted back then in a company such as Bosch, where there are so many units who want to try and work with vision language models and integrate before integrating them into their own autonomous driving solution pipelines, from perception, from prediction, from planning, or from, like, developing hardwares with native capabilities and stuff. Okay, you know what I mean? Let's have a narration over this. This is good.

## Assistant

**Yes. This gives the MVP a clear purpose rooted in your experience: helping multiple engineering teams adapt and evaluate models before integrating them into their products.** DriveLM supplies public data for demonstrating that workflow.

Here is a first interview narrative based on what you’ve described:

> At Bosch, the motivation for the platform came from working with groups developing autonomous-driving applications. Different teams wanted to adapt vision-language models for different purposes—supporting perception, prediction, or planning pipelines, or exploring capabilities that could eventually run on embedded camera hardware.
>
> The common need was a repeatable way to take a model, adapt it to a team’s data, evaluate the result, and decide whether it was suitable for further integration. That motivated my work on a shared post-training platform.
>
> In that environment, the tenants were internal engineering groups. They shared infrastructure, but had different datasets, workloads, and deployment requirements. A team running a small adaptation experiment had different needs from a team processing larger visual inputs or submitting a series of experiments. That made resource management, reproducibility, and evaluation central platform concerns.
>
> I’m now building a compact demonstration of that workflow using public datasets, because I cannot use Bosch’s internal data. DriveLM gives me driving-scene question-answering examples that are relevant to the original application domain.
>
> The MVP lets me make the engineering decisions visible: how jobs enter the platform, how different teams share GPU capacity, how training progress and artifacts are tracked, and how an adapted model is evaluated. I can compare scheduling policies and resource configurations using measured waiting times, GPU memory, training throughput, and completion times.
>
> The connection to deployment is also important. Successful training is one milestone; the next question is whether the resulting model meets the quality and hardware constraints of the intended application.

**The strongest Bosch-specific themes to develop are these.** These are candidate emphases—we should identify which you actually implemented, which you influenced, and which belong to the new MVP.

| Platform concern | Why it matters in the setting you described | Evidence to discuss if you have it |
|---|---|---|
| **Shared workflow across teams** | Groups need similar training infrastructure despite different applications | Common interfaces, reusable training components, onboarding process |
| **Dataset ownership and access** | Data may belong to particular projects or partnerships | Access controls, dataset registration, separation of outputs |
| **GPU allocation and fairness** | Large experiments can delay smaller teams’ iterations | Queue policy, quotas, admission decisions, utilization measurements |
| **Reproducibility and lineage** | Teams need to know exactly what produced a model | Dataset/model versions, configurations, checkpoints, experiment tracking |
| **Application-specific evaluation** | Perception, prediction, and planning require different success criteria | Evaluation suites, regression checks, acceptance thresholds |
| **Hardware-aware handoff** | A useful model must also fit its intended execution environment | Model size, target-device memory, latency, export or compression workflow |

One distinction will strengthen the technical story: **training-GPU metrics describe the platform’s efficiency; target-device metrics describe the model’s deployability.** Measuring training VRAM does not establish that the resulting model fits an embedded camera. The platform can record both, with deployment measurements coming from a separate evaluation stage.

Before we make the historical narrative more specific: **at Bosch, which parts did you personally build—training orchestration, data handling, evaluation, experiment tracking, resource scheduling, or model export?** And which tools did you actually use there? That will let us turn this into a precise ownership story without accidentally attributing today’s Ray/TRL implementation to the earlier platform.

---

## 9. User

Um. Does reinforcement learning post-training still fit into this, or is it only like SFT? For instance, what kind of a group wants to post-train a visual language model with PPO or GRPO or DPO or SFT? SFT is a bit more clear because it gives you—I mean, we can always say it gives a specific structure to how the answer is produced. But let's imagine what kind of an engineering team that develops perception or prediction planning, or, like, things about hardware integration of these models, think of—or need, really, not think of, but need PPO, GRPO type of fine-tuning or post-training.

## Assistant

**Yes—RL post-training fits, especially when a team can score the quality of a generated decision or trajectory.** The need comes from the objective and available feedback, rather than simply from using a VLM.

Also, **SFT teaches much more than output structure**: it can teach scene interpretation, hazard recognition, prediction, and planning behavior through demonstrations.

Think of a scene where a pedestrian approaches a crossing:

| Method | What the team provides | What training encourages |
|---|---|---|
| **SFT** | Scene + expert answer: “Slow down and yield…” | Produce responses or actions resembling expert demonstrations |
| **DPO** | Scene + preferred and rejected outputs | Prefer the better response—for example, a grounded explanation over one inventing a pedestrian’s intention |
| **GRPO** | Scene + a function that scores sampled outputs | Generate several candidates and increase the likelihood of higher-scoring ones |
| **PPO** | Sampled outputs/actions + reward feedback, typically with a learned value estimator | Improve expected reward through policy updates |

DPO is preference optimization; it does not require the online generation-and-reward loop used in typical PPO/GRPO post-training.

**Which teams would actually have a reason to use them?**

**1. A perception or scene-understanding team**

Their problem might be: *“The model describes visible objects correctly most of the time, but sometimes invents hazards or overlooks a critical object.”*

- **SFT:** learn from annotated images and correct descriptions.
- **DPO:** learn from expert-ranked descriptions—grounded, relevant descriptions preferred over hallucinated or incomplete ones.
- **GRPO:** useful if outputs contain objectively checkable fields, such as object categories, counts, or spatial relationships.

For free-form descriptions, defining a reliable reward is harder. An automated judge’s approval is not necessarily evidence that the scene was understood correctly.

**2. A prediction team**

Their problem might be: *“We need plausible future behavior predictions, not just convincing explanations.”*

If the model produces structured future positions or maneuver predictions, the team can score them against recorded futures, using trajectory error or maneuver correctness. That makes reward-based optimization possible.

But **ordinary supervised training may already solve this well**. RL becomes more attractive when the team wants to optimize a non-differentiable score or several interacting criteria. One recorded future also does not represent every plausible future.

**3. A planning team—the clearest RL use case**

Their problem might be:

> “Several actions are plausible. We want actions that make progress while respecting collision, road-boundary, and comfort constraints.”

The model generates candidate maneuvers or trajectories. A simulator or trajectory evaluator scores their consequences. GRPO can compare candidates for a given scene; PPO can support policy optimization using reward feedback across interactions.

There is published precedent: **AlphaDrive combines SFT with GRPO-based planning rewards**, and **Drive-R1 applies reinforcement learning to connect driving reasoning with motion planning**. These are research examples, not evidence of production deployment.

A key distinction: **rewarding the text “I will yield safely” is different from evaluating whether the generated trajectory actually yields safely.**

**4. An embedded-hardware team**

Their primary problem is usually memory, latency, power, and preserving quality after compression. **That does not inherently require PPO or GRPO.**

SFT/distillation, quantization, and deployment optimization may be the more direct tools. Reward-based training could be relevant if the model controls something measurable—such as how many reasoning steps or tool calls it uses—but shorter answers alone do not guarantee acceptable device latency.

For your platform narrative, the credible motivation is:

**Different internal teams need different forms of adaptation because their supervision differs: demonstrations, preferences, or executable rewards. The shared platform manages their data, resources, runs, evaluation, and artifacts.**

For the MVP, **DriveLM SFT is a sound starting point**. A later driving-GRPO experiment needs an explicit reward design: a carefully selected, verifiable QA task would be manageable. DriveLM’s free-form reference answers alone do not give us a simulator-based planning reward.

---

## 10. User

In their work they just evaluate the VLMs using DriveLM dataset right ? Do they do any post training fine tuning ? [https://drive-bench.github.io/](https://drive-bench.github.io/)

## Assistant

**Mostly evaluation, yes—but they also report fine-tuning a DriveLM-Agent baseline.**

Two distinctions:

- **DriveBench is their benchmark**, constructed from DriveLM with rebalanced examples and clean, corrupted, and text-only input conditions.
- **For most VLMs**, they evaluate existing official checkpoints. However, Section 4.1 reports fine-tuning **LLaMA-Adapter-V2 on DriveLM-nuScenes**, using eight A800 GPUs for four epochs, to obtain the DriveLM-Agent baseline. They do not present PPO/GRPO/DPO training as their contribution.

So this paper is primarily about **whether existing and driving-fine-tuned VLMs actually rely on visual evidence**, rather than introducing a new post-training method.

For our MVP, the useful combination would be: **fine-tune on DriveLM training data, then evaluate using DriveBench-style tests**, checking scene overlap before treating any benchmark examples as held out. That lets us ask whether fine-tuning improves actual visual understanding—or mainly teaches the model the expected answer style.

---

## 11. User

So how do we do fine-tuning with Strive LM data, training data? So we take the known VLM, like Qwen or something, and then divide the Strive LM data into training and evaluation, and then use some of it for fine-tuning, post-training, and then use the rest for evaluation. Is that what you want us to do?

## Assistant

**Yes—exactly. With DriveLM, that would be supervised fine-tuning (SFT), using LoRA to update a small subset of parameters.**

The workflow is:

1. **Start with a pretrained VLM**, such as a suitable Qwen-VL model.
2. **Prepare separate training, validation, and test data.** Prefer DriveLM’s existing splits where labels are available. Keep entire driving scenes together: questions about the same image—or nearby frames from the same scene—must not cross splits.
3. **Evaluate the original model on the test set** to establish a baseline.
4. **Fine-tune on the training set.** Each example provides:
   - **Input:** camera images + question.
   - **Training target:** the dataset’s reference answer.

   The model learns to predict that answer; with LoRA, the base weights remain frozen while adapter parameters are trained.
5. **Use validation data** to monitor training and choose the checkpoint.
6. **Evaluate the selected checkpoint on the same test set**, with the same prompts, generation settings, and scoring procedure as the baseline.

During evaluation, **the model sees the images and question, but not the reference answer**. We use that answer afterward for scoring.

For our MVP, we can start with a few hundred training examples and a small, scene-separated evaluation set. This gives us two independent results:

| Model adaptation | Platform performance |
|---|---|
| Did held-out answer quality improve? | How much VRAM and RAM did training use? |
| Does the model use the visual input? | What were throughput, queue time, and completion time? |

**Then we submit separate training jobs under different tenant IDs**, each producing its own adapter. That turns this basic fine-tuning workflow into our multi-tenant platform experiment.

---

## 12. User

Is fine-tuning for SFT on a VLM is different than a pure text-based LLM?

## Assistant

**The training principle is the same. The main difference is how we prepare the inputs and which model components we train.**

Both use next-token prediction: given the input and preceding answer tokens, predict the next token of the reference answer.

| | Text-only LLM SFT | VLM SFT |
|---|---|---|
| **Example input** | Question or conversation | Images + question or conversation |
| **Preprocessing** | Tokenizer converts text into tokens | Processor prepares images and tokenizes text |
| **Model computation** | Text passes through the language model | Vision components encode images; the language model uses visual information to produce text |
| **Training target** | Reference answer text | Reference answer text |
| **Loss for our QA setup** | On answer tokens | On answer tokens—not on image pixels |
| **Resource demands** | Depend on model, sequence length, and batch size | Also depend on image count, resolution, and resulting visual tokens |

For a DriveLM example:

**Input:** camera images + “What should the ego vehicle do?”<br>
**Target answer:** “Slow down and yield to the pedestrian.”

During training, we feed the reference answer too, with causal masking. At the position where the model should predict **“yield,”** it can use the images, the question, and preceding answer words. The loss pushes it toward the reference token.

**LoRA works in both cases.** For a VLM, we additionally decide which components to adapt:

- **Language-model layers:** adapt how the model uses its existing visual representations.
- **Vision-language connector/projector:** adapt how visual features are passed to the language model, where the architecture has one.
- **Vision encoder:** adapt visual feature extraction itself.

We do not have to train all three. **A reasonable MVP starting point is LoRA on the language-model layers, with the vision encoder frozen.**

So your existing SFT workflow carries over. The extra implementation work is **loading images, using the model’s multimodal processor/chat format, batching images correctly, and masking labels so the loss applies to the answer**. The optimizer, job submission, checkpointing, and metrics infrastructure follow the same overall pattern.

---

## 13. User

Can I do the extra implementation work : loading images, using the model’s multimodal processor/chat format, batching images correctly, and masking labels so the loss applies to the answer with Ray Data ? I remember they mentioned in the AnyScale presentation  something like dataloader workers. Or is it something I need to implement myself

## Assistant

**Yes, Ray Data can handle part of it—but you don’t need to implement everything yourself.** Ray Data manages data movement and parallel preprocessing; Hugging Face’s processor and TRL handle model-specific preparation.

I overstated the amount of custom work earlier: **current TRL already supports VLM SFT and provides a multimodal data collator.** Compatibility with your installed version still needs checking.

| Work | Who handles it? | What we implement |
|---|---|---|
| Read images and annotations | Hugging Face Datasets, a PyTorch dataset, or Ray Data | Resolve DriveLM image paths and match images to questions |
| Convert DriveLM records into conversations | Our preprocessing function | Map images, question, and reference answer into the expected schema |
| Resize/normalize images and tokenize text | The VLM’s **processor** | Configure image limits and use the correct processor |
| Assemble padded model-ready batches | TRL’s **multimodal collator**, which calls the processor | Customize only if the built-in path doesn’t support our setup |
| Apply answer-only loss masking | Supported TRL configuration/template or a custom collator | Verify that prompt, image, and padding positions are excluded |
| Parallelize preprocessing and feed training workers | **Ray Data**, if used | Define transformations and connect its output to training |

TRL’s answer-only masking support depends on dataset format, chat template, and library version; simply masking image and padding tokens does **not** necessarily mask the user’s question.

**“Dataloader workers” usually means something more specific.**

A GPU training process needs a steady stream of prepared batches. PyTorch DataLoader workers are **CPU subprocesses** that fetch examples and perform preparation, potentially including collation, while the GPU works on the previous batch.

For example:

- **One training worker:** runs the model on one GPU.
- **Four dataloader workers:** help that training worker read and prepare data.
- They do **not** represent four tenants or four training jobs.

Ray Data is an alternative way to build a scalable input pipeline: it can distribute preprocessing and stream data to training workers. Ray Train supports both native framework loading and Ray Data pipelines.

**For our first MVP, I’d use:**

**Ray Train to launch the job; Hugging Face Datasets + the VLM processor + TRL’s collator inside the training worker.**

Our main custom code would be the **DriveLM-to-training-example conversion**, plus a check of one batch’s images and labels. We can add Ray Data when profiling shows that loading or preprocessing leaves the GPU waiting. It isn’t required just because we’re using Ray Train, and connecting a Ray Data iterator to `SFTTrainer` requires integration rather than passing it in as an ordinary Hugging Face dataset.

That also gives you another measurable platform experiment later: **does more CPU-side data preparation reduce GPU idle time enough to justify its RAM and CPU cost?**

---

## 14. User

Also check the DriveBench codebase. Do they provide anything for data processing ? [https://github.com/worldbench/DriveBench/tree/main/tools](https://github.com/worldbench/DriveBench/tree/main/tools)

## Assistant

**Yes—and they actually use Ray Data in their inference pipeline.** I inspected the code: there are useful pieces to reuse, although it is an evaluation pipeline rather than an SFT data pipeline.

| File | What it does | Value for our MVP |
|---|---|---|
| [tools/preprocess.py](https://github.com/worldbench/DriveBench/blob/main/tools/preprocess.py) | Extracts camera names from question references, such as `<c1,CAM_BACK,…>`, and filters `image_path` to those cameras. If none are referenced, it leaves the images unchanged. | Reusable camera-selection logic |
| [inference/utils.py](https://github.com/worldbench/DriveBench/blob/main/inference/utils.py) | Updates the prompt to describe the supplied cameras; includes JSON loading/saving helpers | Useful camera-aware prompt preparation |
| [inference/llava1.5.py](https://github.com/worldbench/DriveBench/blob/main/inference/llava1.5.py) | Reads JSON with **Ray Data**, loads images with PIL, resizes them, constructs multimodal prompts, and runs vLLM through `map_batches` | A concrete reference for distributed batch evaluation |
| [docs/DATA_PREPAER.md](https://github.com/worldbench/DriveBench/blob/main/docs/DATA_PREPAER.md) | Provides download links, directory layout, and preprocessing command | Helps organize the benchmark assets |

**An important detail about their Ray implementation:** image loading and resizing happen *inside the GPU inference actor’s batch function*. They are not using a separate pool of CPU dataloader workers for that preparation.

For **our SFT pipeline**, we still need to add:

- Conversion into the trainer’s **images + user question + assistant reference answer** format.
- The chosen VLM’s processor and chat template.
- Training-batch collation and answer-only label masking.
- Scene-separated training/validation/test selection.

**We should reuse their data conventions and camera-handling ideas, but use Qwen’s own processor and template.** Their script hardcodes LLaVA-style prompts and a `224 × 224` image resize; those should not be copied directly into our Qwen training setup.

One detail I would fix during adaptation: their prompt helper describes cameras in a canonical order, while image loading follows the dictionary’s value order. **We should explicitly order both together**, so the camera labels always match the images.

This gives us a useful starting point for the **evaluation side** of the platform, plus a small amount of reusable preparation logic for training.

---

## 15. User

is this an overkill for my interview preparation ?

## Assistant

**Yes—the full VLM training + DriveBench evaluation + multi-tenant scheduling scope is overkill for this interview.** We’ve expanded a useful MVP into several projects.

Your main preparation gap is **platform engineering**. You already have substantial autonomous-driving and ML experience. The highest-value work now is demonstrating that you can connect training to a reliable service.

I’d draw the finish line here:

- Submit a job through the API; persist its tenant, configuration, and status.
- Execute your **existing SFT-LoRA training** through Ray.
- Save the adapter and its lineage.
- Record queue time, execution time, peak GPU memory, and throughput.
- Submit jobs from two tenants and demonstrate queue ordering and separate outputs.
- Show how a failed job is recorded and handled.

**Use a modest public text dataset to replace the four examples. One training method is enough.** A simple held-out evaluation completes the workflow.

DriveLM remains a strong domain example for explaining *why* Bosch teams would need this platform. Implementing its multimodal pipeline can come after the core path works. DriveBench, additional RL methods, concurrent GPU sharing, and sophisticated scheduling can wait.

**A small working platform whose decisions you can defend will serve you better than a broad prototype you’re still debugging.** We have enough research now—let’s finish the core workflow and spend the remaining preparation time explaining its trade-offs and practicing interview questions.

---

## 16. User

ok lets pause the DriveLM for now. Lets go back to the platform layers we discussed yesterday starting with customer interface and lets build one full stack with 7 layers and onboard two tenants both with super simple SFT fine-tuning. Then lets discuss evaluation/monitoring/observability of the platform a bit and then move to the DriveLM applications.

## Assistant

**Agreed. We’ll finish one complete path with two tenants and simple SFT-LoRA jobs, then add platform evaluation and observability. DriveLM comes afterward.**

Let’s use this seven-layer structure. These are logical responsibilities; we don’t need seven separate services.

| Layer | Responsibility | MVP implementation |
|---|---|---|
| **1. Customer interface** | Submit jobs and inspect results | Simple Python client or API docs |
| **2. API and authentication** | Identify the tenant and validate requests | FastAPI; one API key per tenant |
| **3. Platform control plane** | Own datasets, job records, configurations, and lifecycle state | Application logic + PostgreSQL |
| **4. Scheduling and orchestration** | Select queued jobs, dispatch them, and reconcile outcomes | Small scheduler loop; FIFO initially |
| **5. Training execution** | Load the model/data and perform SFT-LoRA | Ray worker running your existing TRL code |
| **6. Artifact storage and lineage** | Save adapters/checkpoints and connect them to their inputs | Persistent storage; tenant/job-specific paths and metadata |
| **7. Evaluation and observability** | Measure model results and platform behavior | Held-out loss, job timings, resource metrics, and structured logs |

**Our two tenants can use the same base model and training code**, with separate registered dataset subsets and separate adapters. That gives us a real ownership boundary without introducing another training implementation.

The first complete workflow will be:

1. Tenant A submits its dataset ID and supported model ID.
2. The API derives the tenant identity from the API key and checks dataset ownership.
3. The control plane creates a `QUEUED` job and returns its ID.
4. The scheduler claims the job and dispatches training through Ray.
5. Training runs and saves an adapter.
6. The job becomes `SUCCEEDED`, with an artifact reference—or `FAILED`, with an error.
7. Tenant A retrieves its status and result. Tenant B follows the same path and cannot access A’s resources.

Initially, **one job uses the GPU at a time**. Both tenants can submit jobs while another is running.

We’ll capture basic timestamps and logs while building, then discuss what to monitor, how to diagnose failures, and how to compare scheduling policies.

**To build on your existing implementation, send the project’s folder tree and the working Ray/TRL training script—or make the repository available here.** We’ll start at layer 1 with the customer’s submit/status/result interface and connect it through all seven layers.

---

## 17. User

where do I implement these - in the same python repo where I have my own post training method implementation - sft, ppo etc ?

## Assistant

**Yes—use the same Python repository for this MVP.** Organize the platform code separately from your training methods within it.

For example:

```text
post_training_platform/
├── client/
│   └── demo.py              # Submit jobs as tenant A or B
├── api/
│   ├── routes.py            # Submit/status/result endpoints
│   └── auth.py              # API key → tenant identity
├── control_plane/
│   ├── jobs.py              # Job lifecycle and validation
│   └── datasets.py          # Dataset registration and ownership
├── database/
│   ├── models.py            # Tenant, dataset, job records
│   └── session.py           # PostgreSQL connection
├── orchestration/
│   ├── scheduler.py         # Choose and claim queued jobs
│   └── ray_dispatcher.py    # Launch and track Ray execution
├── training/
│   ├── worker.py            # Common training entry point
│   └── methods/
│       ├── sft.py           # Your existing SFT implementation
│       └── ppo.py           # Your existing PPO implementation
├── artifacts/
│   └── storage.py           # Save/find adapters and checkpoints
├── evaluation/
│   └── evaluate.py
├── observability/
│   └── logging.py
└── pyproject.toml
```

These are **code modules, not separate microservices**. Also, the seven logical layers don’t have to correspond to exactly seven folders.

**Same repository does not mean same running process:**

| Process | Where it runs | What it does |
|---|---|---|
| API server | CPU machine | Receives requests and returns job IDs |
| Scheduler | CPU machine; can initially be the same machine | Reads the queue and dispatches jobs |
| Ray training worker | GPU machine | Imports and executes your SFT method |

The API should **enqueue the job and return**, rather than run `trainer.train()` inside the request handler.

Your existing training code becomes the execution component: it accepts a job configuration and dataset location, performs training, and returns an artifact location and metrics. **We build the platform around it.**

---

## 18. User

Tell me it is easier than I imagine in my head. I am procrastinating.

## Assistant

**It is easier than the picture we’ve built in your head, Gonca.** I made it look bigger by showing you an entire architecture and a dozen folders. You don’t need to create all of that before anything works.

You already ran training. Now we build around something that works, one small piece at a time.

**Your next task is only this: open your existing repo and show me its folder structure.** No architecture decisions, no new datasets, no seven-layer implementation all at once.

The layers are a way to organize the work—not seven mountains you have to climb today. 💙

---

## 19. User

So the real work is in platform control application logic, right? Let's see what I do in the platform control plane. It's application logic. And then what I send is application logic, what I send to scheduling and orchestration in the fourth layer, which is something. Oh sorry, it's in the scheduling and orchestration. How I select queued jobs, dispatch, and reconcile outcomes. It's the real meat layer 4. It's the scheduler I need to implement. Is that the case?

## Assistant

**Yes—layer 4 is a central piece you need to implement, but layers 3 and 4 together contain the platform’s core application logic.** Ray handles execution; it doesn’t automatically implement your customer-level job policy.

The distinction is:

| Layer | Question it answers | Example |
|---|---|---|
| **3. Control plane** | **Is this job valid, and what is its recorded state?** | Tenant A owns the dataset; configuration is valid; create job `123` as `QUEUED` |
| **4. Scheduling and orchestration** | **Which job runs next, how do we launch it, and what happened?** | Claim job `123`; submit it to Ray; track execution; record success or failure |

**Layer 3 doesn’t send “application logic” to layer 4. It stores a job record.** Layer 4 reads that record and acts on it.

For our MVP, your scheduler/orchestrator is essentially a loop:

1. **Check capacity:** is our single training slot free?
2. **Select and claim:** take the oldest eligible queued job, atomically, so it cannot be dispatched twice by competing schedulers.
3. **Dispatch:** pass its configuration to the Ray training entry point and record the execution ID.
4. **Reconcile:** check Ray’s execution state and the saved result; update the database.
5. **Continue:** once the slot is free, launch the next job.

“Reconcile” means **bringing our database’s view into agreement with what actually happened**. For example: the database says `RUNNING`, but Ray reports that the worker failed—so we record `FAILED` and the error.

There are also **two scheduling decisions**:

- **Your platform:** which tenant’s job gets admitted next?
- **Ray:** where can the admitted work execute given its requested resources?

**So yes: a small scheduler loop is the next substantial piece—not a GPU scheduler built from scratch.** FIFO, one active job, and explicit success/failure handling are enough for our first version.

---

## 20. User

save this entire conversation as a markdown for me to download

---

## Source links referenced during the conversation

- [Ray: RL Post-Training using Hugging Face TRL with GRPO](https://docs.ray.io/en/latest/train/examples/transformers/transformer_reinforcement_learning/README.html)
- [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- [Bitext dataset description](https://www.bitext.com/blog/free-customer-support-dataset/)
- [ChartQA dataset card](https://huggingface.co/datasets/HuggingFaceM4/ChartQA/blob/main/README.md)
- [DriveLM official repository](https://github.com/OpenDriveLab/DriveLM)
- [DriveLM challenge and training walkthrough](https://github.com/OpenDriveLab/DriveLM/blob/main/challenge/README.md)
- [LingoQA official repository](https://github.com/wayveai/LingoQA)
- [LingoQA paper](https://wayve.ai/wp-content/uploads/2024/04/2312.14115.pdf)
- [Dolphins / VLM-Driver](https://vlm-driver.github.io/)
- [DriveLMM-o1 repository](https://github.com/ayesha-ishaq/DriveLMM-o1)
- [DriveLMM-o1 dataset card](https://huggingface.co/datasets/ayeshaishaq/DriveLMMo1/blob/main/README.md)
- [AlphaDrive paper](https://arxiv.org/abs/2503.07608)
- [Drive-R1 paper](https://arxiv.org/abs/2506.18234)
- [DriveBench project](https://drive-bench.github.io/)
- [DriveBench paper](https://arxiv.org/html/2501.04003v1)
- [DriveBench code](https://github.com/worldbench/DriveBench)
- [TRL SFT Trainer documentation](https://huggingface.co/docs/trl/sft_trainer)
- [Ray Train data loading and preprocessing](https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html)
