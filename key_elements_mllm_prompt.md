# Key Elements for Designing MLLM Prompts

This guide provides strategies for designing effective prompts for multimodal language models (MLLMs) in visual reasoning tasks.

## 1. Task Definition
**Design Principle:** Your prompt should begin with a clear, unambiguous definition of what the model is expected to accomplish. This establishes the scope and goals of the interaction.

**Implementation Strategy:** Define the specific visual reasoning task, input format, and expected output format. For example:

```
Look at the image containing geometric shapes arranged in a grid. I'll give you a referring expression that describes one of the shapes. Your task is to identify the exact shape being referred to and provide its bounding box coordinates as percentages of the image dimensions [x_min%, y_min%, x_max%, y_max%].
```

## 2. Context about the Scene Structure
**Design Principle:** When working with zero-shot learning scenarios, providing context about the visual domain helps the model understand what to look for without prior training on your specific dataset.

**Implementation Strategy:** Describe the key visual elements, their properties, and spatial arrangement that the model should recognize. For a synthetic shapes dataset, you might include:

```
The image contains shapes (triangles, squares, or circles) of different sizes (small or large), colors (red, blue, green, yellow, etc.), and styles (solid-filled, two-color split, or outlined). These shapes are arranged in an invisible grid, with each shape placed in a cell. The grid might be 2x2, 3x3, 4x4, or larger.
```

## 3. Output Format with Examples
**Design Principle:** MLLMs perform better when given explicit instructions about the expected output format. Including examples demonstrates the structure and level of detail required in responses.

**Implementation Strategy:** Specify the exact format you want the model to use, including any JSON structure, required fields, and data types. Always include at least one concrete example. For instance:

```
Provide your answer in this format:
{
  "bounding_box": [x_min%, y_min%, x_max%, y_max%],
  "confidence": (value between 0-1 reflecting your certainty),
  "explanation": "Brief description of why you selected this shape"
}
```

For example:
```
{
  "bounding_box": [25.5, 30.2, 35.7, 42.8],
  "confidence": 0.95,
  "explanation": "This is the large red triangle in the second row, third column"
}
```

## 4. Handling Edge Cases
**Design Principle:** Robust prompts should anticipate potential edge cases and provide clear guidance on how the model should respond in ambiguous situations.

**Implementation Strategy:** Explicitly instruct the model on how to handle cases where the referring expression is ambiguous, non-existent, or could apply to multiple objects. Include specific output formats for these scenarios. For example:

```
If the referring expression doesn't match any shape in the image, return empty coordinates with zero confidence and explain why no match was found. If multiple shapes could match the referring expression, select the best candidate and indicate lower confidence in your response, explaining the ambiguity.
```

## Conclusion
Effective MLLM prompts for visual reasoning tasks should clearly define the task, provide context about visual elements, specify output formats with examples, and address potential edge cases. By following these design principles, you can create prompts that maximize model performance even in zero-shot scenarios.
