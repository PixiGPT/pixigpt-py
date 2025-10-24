"""Example using PixiGPT vision and moderation APIs."""

import os
from dotenv import load_dotenv
from pixigpt import (
    Client,
    VisionAnalyzeRequest,
    VisionTagsRequest,
    VisionOCRRequest,
    VisionVideoRequest,
    ModerationTextRequest,
    ModerationMediaRequest,
)

# Load environment
load_dotenv()

# Create client
client = Client(
    api_key=os.getenv("PIXIGPT_API_KEY"),
    base_url=os.getenv("PIXIGPT_BASE_URL"),
)


def main():
    """Run vision and moderation examples."""

    # Example 1: Image Analysis
    print("=== Image Analysis ===")
    response = client.analyze_image(
        VisionAnalyzeRequest(
            image_url="https://rub.soulkyn.com/b6e4b172-b8d8-4552-9361-abd56a1e84ee.webp",
            user_prompt="Describe this image in detail.",
        )
    )
    print(f"Analysis: {response.result}")
    print(f"Tokens: {response.usage.total_tokens}\n")

    # Example 2: Tag Generation
    print("=== Tag Generation ===")
    response = client.analyze_image_for_tags(
        VisionTagsRequest(
            image_url="https://rub.soulkyn.com/b6e4b172-b8d8-4552-9361-abd56a1e84ee.webp"
        )
    )
    print(f"Tags: {response.result}")
    print(f"Tokens: {response.usage.total_tokens}\n")

    # Example 3: OCR Text Extraction
    print("=== OCR Text Extraction ===")
    response = client.extract_text(
        VisionOCRRequest(
            image_url="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vl_4b_8b_text_instruct.jpg"
        )
    )
    print(f"Extracted text: {response.result}")
    print(f"Tokens: {response.usage.total_tokens}\n")

    # Example 4: Video Analysis
    print("=== Video Analysis ===")
    response = client.analyze_video(
        VisionVideoRequest(
            video_url="https://rub.soulkyn.com/d8d917b0-bb37-43d6-9d3f-19eee1547065.mp4",
            user_prompt="Describe what happens in this video.",
        )
    )
    print(f"Video analysis: {response.result}")
    print(f"Tokens: {response.usage.total_tokens}\n")

    # Example 5: Text Moderation
    print("=== Text Moderation ===")
    response = client.moderate_text(
        ModerationTextRequest(
            prompt="Generate a beautiful landscape with mountains and sunset"
        )
    )
    print(f"Category: {response.category} (score: {response.score})")
    print(f"Tokens: {response.usage.total_tokens}\n")

    # Example 6: Image Moderation
    print("=== Image Moderation ===")
    response = client.moderate_media(
        ModerationMediaRequest(
            media_url="https://rub.soulkyn.com/b6e4b172-b8d8-4552-9361-abd56a1e84ee.webp",
            is_video=False,
        )
    )
    print(f"Category: {response.category} (score: {response.score})")
    print(f"Tokens: {response.usage.total_tokens}\n")

    # Example 7: Video Moderation
    print("=== Video Moderation ===")
    response = client.moderate_media(
        ModerationMediaRequest(
            media_url="https://rub.soulkyn.com/d8d917b0-bb37-43d6-9d3f-19eee1547065.mp4",
            is_video=True,
        )
    )
    print(f"Category: {response.category} (score: {response.score})")
    print(f"Tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
