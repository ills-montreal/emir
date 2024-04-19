## Computer Vision Embeddings

    micromamba create -n emir python==3.10 -c conda-forge
    micromamba activate emir
    pip install numpy==1.26.4
    pip install torch==2.2.2
    pip install torchvision==0.17.2
    pip install transformers==4.40.0
    pip install datasets==2.19.0
    pip install pillow==10.3.0

    Make a folder using the huggingface model's names/folders and datasets like:
        Model folder/model name/dataset folder/dataset_name/split/{embeddings.npy|labels.npy}
        
        (Embeddings.npy just being a torch.save or numpy.save of the tensor of shape (n_samples, dim) 