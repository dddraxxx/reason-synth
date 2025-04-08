import re
import numpy as np
from scipy.optimize import linear_sum_assignment

# Define the IoU function directly in this test file
def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union

# Define the multi_bbox_iou_reward function directly here for testing
def multi_bbox_iou_reward(completions, solution, iou_threshold_low=0.1, iou_threshold_high=0.9, **kwargs):
    """Extract multiple bounding boxes from completions and compute IoU rewards against solution.
    Uses bipartite matching to find optimal assignment between predicted and ground truth boxes.

    Args:
        completions: List of completions, where each completion is a list of messages
        solution: List of lists of ground truth bounding boxes, where each inner list is a list of bbox coordinates
                 [x_min, y_min, x_max, y_max] for each sample
        iou_threshold_low: IoU values below this threshold will be set to 0
        iou_threshold_high: IoU values above this threshold will be set to 1
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    matching_info = []

    for content, gt_boxes in zip(contents, solution):
        reward = 0.0
        match_info = {
            "pred_boxes": [],
            "gt_boxes": gt_boxes,
            "matches": [],
            "match_ious": [],
            "reward": 0.0
        }

        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if not content_answer_match:
                rewards.append(reward)
                matching_info.append(match_info)
                continue

            content_answer = content_answer_match.group(1).strip()

            # Check for "no match" response
            if "not exist" in content_answer.lower():
                # If ground truth also has no boxes, this is correct
                if len(gt_boxes) == 0:
                    reward = 1.0
                match_info["pred_boxes"] = "not exist"
                match_info["reward"] = reward
                rewards.append(reward)
                matching_info.append(match_info)
                continue

            # Find all bounding boxes in the answer - now using re.DOTALL
            all_bbox_matches = re.findall(bbox_pattern, content_answer, re.DOTALL)
            if not all_bbox_matches:
                # No boxes found in prediction
                # if len(gt_boxes) == 0:
                #     reward = 1.0  # Correctly predicted no boxes
                match_info["reward"] = reward
                rewards.append(reward)
                matching_info.append(match_info)
                continue

            # Convert matches to lists of coordinates
            pred_boxes = []
            for match in all_bbox_matches:
                try:
                    box = [int(float(match[0])), int(float(match[1])), int(float(match[2])), int(float(match[3]))]
                    pred_boxes.append(box)
                except ValueError:
                    continue  # Skip malformed boxes

            match_info["pred_boxes"] = pred_boxes

            # Handle special cases
            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                reward = 1.0  # Both prediction and ground truth have no boxes
                match_info["reward"] = reward
                rewards.append(reward)
                matching_info.append(match_info)
                continue

            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                # One has boxes, the other doesn't
                match_info["reward"] = reward
                rewards.append(reward)  # reward stays 0
                matching_info.append(match_info)
                continue

            # Create IoU matrix for bipartite matching
            iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
            raw_iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))  # Store raw IoU values before thresholding

            for i, pred_box in enumerate(pred_boxes):
                for j, gt_box in enumerate(gt_boxes):
                    raw_iou = iou(pred_box, gt_box)
                    raw_iou_matrix[i, j] = raw_iou

                    # Apply IoU thresholding
                    this_iou = raw_iou
                    if this_iou < iou_threshold_low:
                        this_iou = 0.0
                    elif this_iou > iou_threshold_high:
                        this_iou = 1.0
                    iou_matrix[i, j] = this_iou

            # Convert to cost matrix (1 - IoU)
            cost_matrix = 1 - iou_matrix

            # Find optimal assignment using Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matched_ious = [iou_matrix[i, j] for i, j in zip(row_indices, col_indices)]
            raw_matched_ious = [raw_iou_matrix[i, j] for i, j in zip(row_indices, col_indices)]

            # Store matching information
            match_info["matches"] = list(zip(row_indices.tolist(), col_indices.tolist()))
            match_info["match_ious"] = raw_matched_ious

            # Calculate reward - only continuous mode now
            if matched_ious:
                # Just use the sum of all IoUs
                reward = sum(matched_ious)

            match_info["reward"] = reward

        except Exception as e:
            print(f"Error in multi_bbox_iou_reward: {e}")
            pass

        rewards.append(reward)
        matching_info.append(match_info)

    return rewards, matching_info

def create_completion(content_text):
    """Helper function to create a completion object in the expected format"""
    return [{"content": content_text}]

def visualize_matching(match_info, test_name=""):
    """Visualize box matching with detailed information"""
    print(f"\n--- {test_name} ---")

    pred_boxes = match_info["pred_boxes"]
    gt_boxes = match_info["gt_boxes"]
    matches = match_info["matches"]
    match_ious = match_info["match_ious"]
    reward = match_info["reward"]

    # Special case for "no match" response
    if pred_boxes == "not exist":
        print("Predicted: 'not exist'")
        print(f"Ground truth boxes: {gt_boxes}")
        print(f"Reward: {reward}")
        return

    print(f"Predicted boxes ({len(pred_boxes)}):")
    for i, box in enumerate(pred_boxes):
        print(f"  P{i}: {box}")

    print(f"Ground truth boxes ({len(gt_boxes)}):")
    for i, box in enumerate(gt_boxes):
        print(f"  G{i}: {box}")

    print("Matching:")
    if not matches:
        print("  No valid matches found")
    else:
        for i, ((pred_idx, gt_idx), iou_val) in enumerate(zip(matches, match_ious)):
            pred_box = pred_boxes[pred_idx] if pred_idx < len(pred_boxes) else "N/A"
            gt_box = gt_boxes[gt_idx] if gt_idx < len(gt_boxes) else "N/A"
            print(f"  Match {i+1}: P{pred_idx} → G{gt_idx} (IoU: {iou_val:.4f})")
            print(f"    Pred box: {pred_box}")
            print(f"    GT box:   {gt_box}")

    print(f"Final reward: {reward:.4f}")

def test_perfect_match():
    """Test case where predicted boxes perfectly match ground truth"""
    gt_boxes = [[10, 20, 30, 40], [50, 60, 70, 80]]

    # Perfect match, same order
    completion_same_order = create_completion("""
    <think>
    I see two shapes in the image that match the description:
    1. A red triangle at [10, 20, 30, 40]
    2. A blue circle at [50, 60, 70, 80]
    </think>
    <answer>
    {
        "bounding_boxes": [[10, 20, 30, 40], [50, 60, 70, 80]]
    }
    </answer>
    """)

    # Perfect match, different order
    completion_diff_order = create_completion("""
    <think>
    I see two shapes in the image that match the description:
    1. A blue circle at [50, 60, 70, 80]
    2. A red triangle at [10, 20, 30, 40]
    </think>
    <answer>
    {
        "bounding_boxes": [[50, 60, 70, 80], [10, 20, 30, 40]]
    }
    </answer>
    """)

    # Test with continuous reward
    _, match_info_same = multi_bbox_iou_reward([completion_same_order], [gt_boxes])
    _, match_info_diff = multi_bbox_iou_reward([completion_diff_order], [gt_boxes])

    visualize_matching(match_info_same[0], "Perfect Match (Same Order)")
    visualize_matching(match_info_diff[0], "Perfect Match (Different Order)")

def test_partial_match():
    """Test case where some boxes match and others don't"""
    gt_boxes = [[10, 20, 30, 40], [50, 60, 70, 80]]

    # One box matches perfectly, one is slightly off
    completion_partial = create_completion("""
    <think>
    I see two shapes in the image that match the description:
    1. A red triangle at [10, 20, 30, 40]
    2. A blue circle at [52, 58, 71, 79]
    </think>
    <answer>
    {
        "bounding_boxes": [[10, 20, 30, 40], [52, 58, 71, 79]]
    }
    </answer>
    """)

    # One box matches, second box completely wrong
    completion_bad_box = create_completion("""
    <think>
    I see two shapes in the image that match the description:
    1. A red triangle at [10, 20, 30, 40]
    2. A blue circle at [100, 100, 120, 120]
    </think>
    <answer>
    {
        "bounding_boxes": [[10, 20, 30, 40], [100, 100, 120, 120]]
    }
    </answer>
    """)

    # Missing one box
    completion_missing = create_completion("""
    <think>
    I see one shape in the image that matches the description:
    1. A red triangle at [10, 20, 30, 40]
    </think>
    <answer>
    {
        "bounding_boxes": [[10, 20, 30, 40]]
    }
    </answer>
    """)

    # Extra box
    completion_extra = create_completion("""
    <think>
    I see three shapes in the image that match the description:
    1. A red triangle at [10, 20, 30, 40]
    2. A blue circle at [50, 60, 70, 80]
    3. A green square at [100, 100, 120, 120]
    </think>
    <answer>
    {
        "bounding_boxes": [[10, 20, 30, 40], [50, 60, 70, 80], [100, 100, 120, 120]]
    }
    </answer>
    """)

    # Test all cases with continuous reward
    _, match_infos = multi_bbox_iou_reward(
        [completion_partial, completion_bad_box, completion_missing, completion_extra],
        [gt_boxes, gt_boxes, gt_boxes, gt_boxes]
    )

    visualize_matching(match_infos[0], "Partial Match - One box perfect, one slightly off")
    visualize_matching(match_infos[1], "Partial Match - One box perfect, one completely wrong")
    visualize_matching(match_infos[2], "Partial Match - Missing one box")
    visualize_matching(match_infos[3], "Partial Match - Extra box")

def test_iou_thresholding():
    """Test the IoU thresholding functionality"""
    # Test with a box that has very low IoU (less than threshold_low)
    low_iou_completion = create_completion("""
    <think>
    I see a shape at [1, 1, 5, 5].
    </think>
    <answer>
    {
        "bounding_boxes": [[1, 1, 5, 5]]
    }
    </answer>
    """)

    # Test with a box that has very high IoU (greater than threshold_high)
    high_iou_completion = create_completion("""
    <think>
    I see a shape at [10, 20, 31, 41].
    </think>
    <answer>
    {
        "bounding_boxes": [[10, 20, 31, 41]]
    }
    </answer>
    """)

    # Test with a box that has medium IoU (between thresholds)
    medium_iou_completion = create_completion("""
    <think>
    I see a shape at [12, 22, 30, 40].
    </think>
    <answer>
    {
        "bounding_boxes": [[12, 22, 30, 40]]
    }
    </answer>
    """)

    gt_boxes = [[10, 20, 30, 40]]

    # Test all cases with thresholding
    _, match_infos = multi_bbox_iou_reward(
        [low_iou_completion, high_iou_completion, medium_iou_completion],
        [gt_boxes, gt_boxes, gt_boxes],
        iou_threshold_low=0.1,
        iou_threshold_high=0.9
    )

    visualize_matching(match_infos[0], "IoU Thresholding - Very Low IoU (< 0.1)")
    visualize_matching(match_infos[1], "IoU Thresholding - Very High IoU (> 0.9)")
    visualize_matching(match_infos[2], "IoU Thresholding - Medium IoU (between thresholds)")

def test_re_dotall_flag():
    """Test that the re.DOTALL flag properly handles newlines in box coordinates"""
    # Box with newlines between coordinates
    newline_completion = create_completion("""
    <think>
    I see a shape at coordinates with newlines.
    </think>
    <answer>
    {
        "bounding_boxes": [[10,
        20,
        30,
        40]]
    }
    </answer>
    """)

    gt_boxes = [[10, 20, 30, 40]]

    # Test with re.DOTALL flag
    _, match_infos = multi_bbox_iou_reward([newline_completion], [gt_boxes])

    visualize_matching(match_infos[0], "re.DOTALL flag - Box with newlines between coordinates")

def test_unequal_boxes():
    """Test handling of unequal numbers of boxes"""
    gt_boxes = [[10, 20, 30, 40], [50, 60, 70, 80], [100, 110, 120, 130]]

    # Fewer boxes in prediction
    fewer_boxes = create_completion("""
    <think>
    I see two shapes in the image.
    </think>
    <answer>
    {
        "bounding_boxes": [[10, 20, 30, 40], [50, 60, 70, 80]]
    }
    </answer>
    """)

    # More boxes in prediction
    more_boxes = create_completion("""
    <think>
    I see four shapes in the image.
    </think>
    <answer>
    {
        "bounding_boxes": [[10, 20, 30, 40], [50, 60, 70, 80], [100, 110, 120, 130], [200, 210, 220, 230]]
    }
    </answer>
    """)

    # Test handling of unequal boxes
    _, match_infos = multi_bbox_iou_reward(
        [fewer_boxes, more_boxes],
        [gt_boxes, gt_boxes]
    )

    visualize_matching(match_infos[0], "Unequal Boxes - Fewer boxes (2 vs 3)")
    visualize_matching(match_infos[1], "Unequal Boxes - More boxes (4 vs 3)")

def test_no_match_cases():
    """Test cases with no matches or no box provided"""
    # Empty ground truth - correctly says "not exist"
    correct_response = create_completion("""
    <think>No matching shapes found.</think>
    <answer>not exist</answer>
    """)

    # Empty ground truth - says something else
    incorrect_response = create_completion("""
    <think>No matching shapes found.</think>
    <answer>nothing</answer>
    """)

    # Test with empty ground truth
    rewards, match_infos = multi_bbox_iou_reward(
        [correct_response, incorrect_response],
        [[], []]  # Empty ground truth boxes
    )

    visualize_matching(match_infos[0], "Empty GT - Correctly says 'not exist'")
    visualize_matching(match_infos[1], "Empty GT - Incorrectly says 'nothing'")

    print(f"\nReward for correct 'not exist': {rewards[0]}")
    print(f"Reward for incorrect 'nothing': {rewards[1]}")
    # assert rewards[0] == 1.0, "Should get reward 1.0 for correctly saying 'not exist'"
    # assert rewards[1] == 0.0, "Should get reward 0.0 for not saying 'not exist' exactly"

if __name__ == "__main__":
    print("COMPREHENSIVE BOX MATCHING AND REWARD TESTING")
    print("============================================")

    test_perfect_match()
    test_partial_match()
    test_iou_thresholding()
    test_re_dotall_flag()
    test_unequal_boxes()
    test_no_match_cases()

    print("\nAll tests completed!")