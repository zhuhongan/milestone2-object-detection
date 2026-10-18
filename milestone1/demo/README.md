# Live classification

Open `../notebooks/milestone1_live_demo.ipynb` in Colab. It contains the complete inference code and final mapping. Supply the selected ConvNeXt-Tiny `best.pt` checkpoint, then upload single-object photos. The notebook verifies the checkpoint hash, epoch and final class mapping before loading weights.

The folder-based alternative is `../notebooks/Milestone1_Drive_Demo.ipynb`. It expects `MyDrive/Assignment_Live_Demo/models/milestone1_convnext_tiny_best.pt`, reads photos from `milestone1/input` and saves results to `milestone1/output` under that same Drive root. It skips unchanged previously processed images. Each teammate needs those files in their own mounted My Drive or must configure the root path for their account.

The checkpoint is epoch 15 from the four-model comparison. SHA-256:
`fd5a80bd163b4c947aaa05d3749f4b7325e78cc6ef4017b2b18f93737c4c4c85`.

The editable helper sources are `live_classification_helpers.py` and `drive_demo_runtime.py`. They are also embedded in the self-contained notebooks. Original photos are read only; inference exports prediction cards, IDs/names, top-five model scores and JSON/CSV. CPU or CUDA inference is supported. The classifier always chooses among the 73 trained IDs and has no unknown-object class. Checkpoints and private Drive files are not stored in Git.
