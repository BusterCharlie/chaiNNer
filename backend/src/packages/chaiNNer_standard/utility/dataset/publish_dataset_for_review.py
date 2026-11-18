from __future__ import annotations

from pathlib import Path

from api import NodeContext
from dataset_review.ingest import publish_dataset
from nodes.properties.inputs import (
    DirectoryInput,
    NumberInput,
    SliderInput,
    TextInput,
)
from nodes.properties.outputs import DirectoryOutput, TextOutput

from .. import dataset_group


def _optional_file(raw: str | None, what: str) -> Path | None:
    if not raw or not raw.strip():
        return None
    path = Path(raw.strip().strip('"'))
    if not path.is_file():
        raise FileNotFoundError(f"{what} {path} does not exist")
    return path


@dataset_group.register(
    schema_id="chainner:utility:publish_dataset_for_review",
    name="Publish Dataset for Review",
    description=[
        "Copies a tagged training dataset into a review set that friends can help check from a phone or browser.",
        "Images are re-encoded as WebP of at most the given size; captions (.txt) and auto-tagger score sidecars (.scores.json / .json) are imported.",
        "Running it again on the same review directory syncs changes and keeps all votes. Start the review server with `python -m dataset_review serve <review dir>`.",
    ],
    icon="BsTags",
    inputs=[
        DirectoryInput("Dataset"),
        DirectoryInput("Review Directory", must_exist=False),
        NumberInput(
            "Max Size", min=0.1, max=16, default=1, precision=2, step=0.25, unit="MP"
        ),
        SliderInput("WebP Quality", min=50, max=100, default=82),
        TextInput("Tag Database CSV", placeholder="danbooru_e621_merged.csv")
        .make_optional()
        .with_docs(
            "Optional booru tag list (`name,category,count,aliases`, as used by the a1111 tag-autocomplete extension). Powers autocomplete and tag categories."
        ),
        TextInput("Categories File", placeholder="categories.json")
        .make_optional()
        .with_docs(
            "Optional hand-made tag→category mapping (JSON or `tag,category` CSV). Overrides the tag database."
        ),
    ],
    outputs=[
        DirectoryOutput(
            "Review Directory", output_type="Directory { path: Input1.path }"
        ),
        TextOutput("Summary"),
    ],
    node_context=True,
    side_effects=True,
)
def publish_dataset_for_review_node(
    context: NodeContext,
    dataset: Path,
    review_dir: Path,
    max_megapixels: float,
    quality: int,
    tag_db: str | None,
    categories: str | None,
) -> tuple[Path, str]:
    summary = publish_dataset(
        dataset,
        review_dir,
        max_pixels=int(max_megapixels * 1_000_000),
        quality=quality,
        tag_database=_optional_file(tag_db, "Tag database"),
        categories_file=_optional_file(categories, "Categories file"),
        progress=lambda done, total: context.set_progress(done / max(total, 1)),
    )
    if summary.images_failed:
        failed = "; ".join(summary.images_failed[:5])
        return review_dir, f"{summary}. Failed: {failed}"
    return review_dir, str(summary)
