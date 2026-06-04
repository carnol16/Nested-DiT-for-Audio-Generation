from data.dataset import get_dataloader

if __name__ == '__main__':
    loader = get_dataloader(
        latent_dir="data/latents/test",
        stats_path="data/stats.json",
        batch_size=4,
    )

    batch = next(iter(loader))
    print(batch.shape)
    print(batch.mean().item())
    print(batch.std().item())

