def generate_memes(product: str, pain: str, num_memes: int = 3):
    """
    Generate meme texts based on product and pain point.
    This will use the fine-tuned model.
    """
    # Placeholder - will be replaced with actual model inference
    examples = [
        f"When you finally decide to use {product} and suddenly feel motivated...",
        f"Motivation? I don't know her... (but {product} helps)",
        f"Opened {product} and instantly wanted to change my life"
    ]
    return examples[:num_memes]

if __name__ == "__main__":
    print(generate_memes("Gym App", "no motivation to train"))