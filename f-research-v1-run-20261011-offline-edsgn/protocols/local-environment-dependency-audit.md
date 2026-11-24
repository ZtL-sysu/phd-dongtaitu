# P2 dependency/device inspection (initial)

Selected local existing Conda environment: dyg-cold (no package/global settings changed). Python3.13.13,PyTorch2.11.0+cu128,CUDA runtime12.8; NVIDIA RTX3060 12288MiB. Snapshot2026-10-11: free9441MiB,util77%; shared GPU work exists, no other processes stopped. CPU12 logical and31.9GiB host RAM. Existing numpy,pandas,scipy,pytest,PyYAML,matplotlib,psutil,pypdf,PyMuPDF available. Set2CPU threads during this project's runs. Actual package freeze and GPU/driver/memory telemetry stored at smoke completion.

Use the Conda environment's own python executable for all scripts. The environment already has required packages, so selection avoids modifying the shared environment. A separate environment is needed only if an external baseline requires incompatible packages; the preferred adapter imports its torch model alone.

AutoResearchClaw source inspection: C:/code/AutoResearchClaw,pyproject version0.3.1,requiresPython>=3.11. Installed source enum maps PAPER_OUTLINE16,PAPER_DRAFT17,PEER_REVIEW18,PAPER_REVISION19,QUALITY_GATE20,EXPORT_PUBLISH22,CITATION_VERIFY23. CLI source inspected rather than running an API/provider smoke. Manuscript production is outside this user's offline request; ARC writing stages have not executed and remain pending. No paid API test/provider assumption is needed for experimental phases. No credential/config secret was printed.

Scientific GPU timing must preserve actual shared-load telemetry and paired/repeated policy replay; this snapshot is feasibility information, not clean dedicated-hardware speed evidence.
