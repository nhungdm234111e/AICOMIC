from textwrap import dedent


def build_prompt(text: str) -> str:
    """Return a fully formatted illustration prompt for the image model."""
    user_text = text.strip()
    return dedent(
        f"""
        You are an AI comic illustrator.
        Generate one high-quality comic panel based on the following scene:

        {user_text}

        Rules:
        - consistent character appearance
        - vibrant anime-comic hybrid style
        - no text inside the image
        - cinematic lighting, bold outlines
        """
    ).strip()