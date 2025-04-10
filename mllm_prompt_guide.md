# MLLM Visual Reasoning Prompts

This guide provides a streamlined approach for prompting multimodal language models on visual reasoning tasks with referring expressions.

## Core Elements

### 1. Task Directive
Direct instruction to find the bounding box for a referring expression.

### 2. Scene Context
Essential information about the visual elements:
```
The image contains shapes (triangles, squares, or circles) of different sizes (small or large), colors (red, blue, green, yellow, purple, orange), and styles (solid-filled, two-color split, or outlined). These shapes are arranged in a grid layout.
```

### 3. Output Structure
Structured format for reasoning and answer:
```
Output your reasoning process within <think> </think> tags and the answer, as JSON, within <answer> </answer> tags.
```

### 4. Edge Cases
Handling of ambiguous situations:
```
If no matching shape is found, return empty coordinates. If multiple shapes match, select the best candidate and explain your choice.
```

## Refined Sample Prompt

```

The image contains shapes (triangles, squares, or circles) of at most two sizes (small or large), colors (red, blue, green, yellow, purple, orange), and at most three styles (solid-filled, two-color split, or outlined).

Please find the corresponding bounding box for {referring_expression}.
Output your reasoning process within <think> </think> tags and the answer, as JSON, within <answer> </answer> tags. The JSON should include the bounding box coordinates [x_min, y_min, x_max, y_max].
If no matching shape is found, return "no match" in answer. If multiple shapes match, output all bounding boxes in answer.
```

## Expected Model Response

```
<think>
I need to find a {referring_expression} in the image.
1. First, I'll scan the image for {referring_expression}
2. Then I'll identify which {referring_expression} are {color}
3. Finally, I'll determine which {referring_expression} is {size}

I can see a {referring_expression} located in the top-left portion of the image.
Its bounding box coordinates as percentages of the image dimensions are approximately [15.2, 12.8, 32.5, 35.7].
</think>

<answer>
{
  "bbox": [15.2, 12.8, 32.5, 35.7],
  "confidence": 0.95
}
</answer>
```
