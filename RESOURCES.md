# Related work

Neighboring work, sources for the incident replays, and the ideas behind the detectors.

## Closest projects

- SLEIGHT-Bench (Anthropic), blind spots in LLM-based monitors: https://alignment.anthropic.com/2026/sleight-bench/ , code: https://github.com/safety-research/sleight-bench
- MonitoringBench, semi-automated red-teaming of agent monitors: https://arxiv.org/abs/2605.09684
- Better Attacks for Better Monitors (ICLR 2026 workshop, earlier version of the above): https://openreview.net/forum?id=gVlwWUgrzq
- agentwatch (adelic-ai), reconciles an agent's transcript against OS audit records: https://github.com/adelic-ai/agentwatch
- agentnorm, runtime behavioral monitor with attack-family evaluation: https://github.com/kaustubhspatil/agentnorm
- agentomaly, runtime behavioral anomaly detector: https://github.com/sushaan-k/agentomaly
- AISI control red team, evolutionary attacks on LLM monitors: https://www.aisi.gov.uk/blog/how-our-new-control-red-team-is-stress-testing-frontier-monitors
- Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols: https://arxiv.org/abs/2510.09462
- Monitoring LLM Agents for Sequentially Contextual Harm: https://openreview.net/pdf?id=PGsM81SWHt
- Diffuse AI Control on Fuzzy Tasks: https://arxiv.org/abs/2606.08892

## Sources for the incident replays

- Hugging Face, Anatomy of a Frontier Lab Agent Intrusion (technical timeline): https://huggingface.co/blog/agent-intrusion-technical-timeline
- SafeDep, Miasma worm write-up: https://safedep.io/miasma-worm-ai-coding-agent-config-injection/

## Digit forensics

- Rightmost-digit checks (Stanford Best Practices in Science): https://bps.stanford.edu/home/statistical-forensics/statistical-forensics-techniques-detect-and-eliminate-fraud/techniques-6
- Charpentier, What Does a Benford Test Actually Test?: https://arxiv.org/abs/2609.18424
- Barabesi et al., Statistical Detection of Adversarial Compliance With Benford's Law: https://doi.org/10.1002/sam.70081

## Real agent traces (for calibrating on more than synthetic data)

- TraceLab, coding-agent traces (paper: https://arxiv.org/abs/2606.30560 , code: https://github.com/uw-syfi/TraceLab , site: https://tracelab.cs.washington.edu/)
- Trace Commons agent traces: https://huggingface.co/datasets/trace-commons/agent-traces

## Evaluation harnesses and settings

- ControlArena (UK AISI and Redwood): https://github.com/UKGovernmentBEIS/control-arena
- BashArena: https://arxiv.org/abs/2512.15688
- LinuxArena: https://arxiv.org/abs/2604.15384
- ASMR-Bench (auditing sabotage in ML research code): https://arxiv.org/abs/2604.16286
- AgentDojo: https://github.com/ethz-spylab/agentdojo
- AgentSight, system-level agent observability: https://github.com/eunomia-bpf/agentsight

## Claude Code

- Hooks reference: https://code.claude.com/docs/en/hooks

## Redundancy and objectivity


- Riedel and Zurek, Quantum Darwinism in an Everyday Environment: Huge Redundancy in Scattered Photons: https://arxiv.org/abs/1001.3419
- Functional Information in Quantum Darwinism: An Operational Measure of Objectivity (functional information = log2 of redundancy): https://arxiv.org/abs/2509.17775

## Weight watermarking

- Uchida et al., Embedding Watermarks into Deep Neural Networks (2017): https://arxiv.org/abs/1701.04082
