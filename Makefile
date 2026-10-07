# Three venvs: incompatible numpy pins (mediapipe / wilor-mini / hamer+chumpy). See `make env`.
R := $(CURDIR)
PY := $(R)/.venv/bin/python
PY_WILOR := $(R)/.venv-wilor/bin/python
PY_HAMER := $(R)/.venv-hamer/bin/python
SMOKE := $(R)/data/cache/smoke
HAMER_COMMIT := 3a01849f4148352e9260b69bf28b65d1671a4905

.PHONY: env smoke smoke-mediapipe smoke-wilor smoke-hamer

env:  ## needs MANO_RIGHT.pkl in mano_data/ (register at mano.is.tue.mpg.de)
	uv venv --python 3.10 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt
	uv venv --python 3.10 .venv-wilor && VIRTUAL_ENV=.venv-wilor uv pip install pip setuptools wheel numpy \
	  && VIRTUAL_ENV=.venv-wilor uv pip install --no-build-isolation --index-strategy unsafe-best-match \
	     --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements-wilor.txt
	mkdir -p third_party/wilor/pretrained_models && cp mano_data/MANO_RIGHT.pkl third_party/wilor/pretrained_models/
	git submodule update --init --recursive third_party/hamer   # pinned to $(HAMER_COMMIT)
	uv venv --python 3.10 .venv-hamer && VIRTUAL_ENV=.venv-hamer uv pip install pip setuptools wheel numpy==1.23.5 \
	  && VIRTUAL_ENV=.venv-hamer uv pip install --no-build-isolation --index-strategy unsafe-best-match \
	     --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements-hamer.txt \
	  && VIRTUAL_ENV=.venv-hamer uv pip install --no-deps -e third_party/hamer
	cd third_party/hamer && test -d _DATA/hamer_ckpts || (wget -c https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz \
	  && tar --warning=no-unknown-keyword --exclude=".*" -xf hamer_demo_data.tar.gz)
	mkdir -p third_party/hamer/_DATA/data/mano && cp mano_data/MANO_RIGHT.pkl third_party/hamer/_DATA/data/mano/

smoke-mediapipe:
	cd infer && $(PY) run_mediapipe.py $(R)/data/samples $(SMOKE)
smoke-wilor:
	cd infer && $(PY_WILOR) run_wilor.py $(R)/data/samples $(SMOKE)
smoke-hamer: smoke-wilor  # HaMeR reuses WiLoR's detector boxes
	cd third_party/hamer && PYOPENGL_PLATFORM=egl PYTHONPATH=$(R)/infer $(PY_HAMER) $(R)/infer/run_hamer.py $(R)/data/samples $(SMOKE) $(SMOKE)
smoke: smoke-mediapipe smoke-hamer
	$(PY) infer/overlay.py data/samples $(SMOKE) $(SMOKE)/overlay.jpg

# ---- P1: DexYCB + synthetic gloves --------------------------------------------------------
DEX := data/raw/dexycb
.PHONY: dexycb meta events leak grid p1
dexycb:  ## calibration + subject-01 (12 GB, Google Drive via gdown); CC BY-NC 4.0
	mkdir -p $(DEX) && cd $(DEX) && test -d calibration || ($(R)/.venv-hamer/bin/gdown 1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ -O calibration.tar.gz && tar -xzf calibration.tar.gz)
	cd $(DEX) && test -d 20200709-subject-01 || ($(R)/.venv-hamer/bin/gdown 1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk -O subject01.tar.gz && tar -xzf subject01.tar.gz)
meta:
	$(PY) capture/make_meta.py
events: meta
	$(PY) capture/events.py
leak: meta
	cd capture && $(PY) mask_leak.py $$($(PY) -c "import csv;print(' '.join(sorted({'../'+r['src_dir'].rsplit('/',1)[0] for r in csv.DictReader(open('../data/meta.csv'))})))") | tee ../report/p1_mask_leak.txt
grid: events
	cd capture && $(PY) glove_grid.py ../report/fig_glove_grid.jpg $$($(PY) -c "import csv;e=list(csv.DictReader(open('../data/events.csv')));print(' '.join(f'../data/raw/dexycb/20200709-subject-01/{r[\"seq\"]}:{r[\"frame\"]}' for r in (e[0],e[9],e[14])))")
p1: meta events leak grid

# ---- P2: render + inference (cached; reruns skip finished groups) -------------------------
FRAMES := $(R)/data/cache/frames.csv
.PHONY: render infer-mediapipe infer-wilor infer-hamer infer
render: meta
	$(PY) capture/render.py
infer-mediapipe:
	cd infer && $(PY) run_mediapipe.py $(FRAMES) 16
infer-wilor:
	cd infer && $(PY_WILOR) run_wilor.py $(FRAMES)
infer-hamer: infer-wilor
	cd third_party/hamer && PYOPENGL_PLATFORM=egl PYTHONPATH=$(R)/infer $(PY_HAMER) $(R)/infer/run_hamer.py $(FRAMES)
infer: infer-mediapipe infer-hamer

# ---- P2: Roboflow real-glove detection check ---------------------------------------------
RF := $(R)/data/raw/roboflow_gloves/test
.PHONY: rf-infer rf-bench
rf-infer:
	cd infer && $(PY_WILOR) run_wilor.py $(RF) $(R)/data/cache/roboflow
	cd infer && $(PY) run_mediapipe.py $(RF) $(R)/data/cache/roboflow
rf-bench:
	$(PY) eval/roboflow_bench.py

# ---- P2: metrics, summary, figures --------------------------------------------------------
.PHONY: metrics summary figures p2
metrics:
	cd eval && $(PY) metrics.py
summary: metrics
	cd eval && $(PY) summary.py
figures: summary rf-bench
	$(PY) report/plots.py
	$(PY) report/examples.py
p2: render infer rf-infer figures

# ---- P3: interventions + gate (cached predictions only) -----------------------------------
.PHONY: p3 pdf all
p3: summary
	cd eval && $(PY) p3.py > out/p3_log.txt && $(PY) recovery_md.py > /dev/null
	$(PY) report/plots.py
pdf: p3
	$(PY) report/build_pdf.py
all: p2 p3 pdf   ## from cached data every inference step is skipped (JSONL cache)
