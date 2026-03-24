`pip install anthropic` は必要ありません．全てモックで置き換えました．実際のAPIを呼び出すことなく実行できるはずです． 

まず `source .venv/bin/activate` を実行し，出来なければ `uv venv && source .venv/bin/activate` をしなさい．

pythonの代わりにuvを使いなさい．つまり `python -m agents.s01_agent_loop` の代わりに `uv run -m agents.s01_agent_loop` としなさい．
