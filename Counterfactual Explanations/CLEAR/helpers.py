class check_sparsity:


    def __init__(self, max_number_of_features):
        self.m = max_number_of_features

    

    def l0_norm(self, vector):
        """
        Calculate the L0 norm of a vector, which is the count of non-zero elements in the vector.

        Args:
        - vector: The input vector (list or array).

        Returns:
        - The L0 norm of the vector (count of non-zero elements).
        """
        count = 0
        for element in vector:
            if element != 0:
                count += 1
        return count
    

    def is_sparse(self,unit,  counterfactual):
        diff = unit - counterfactual
        if self.l0_norm(diff) < self.m:
            return True
        else:
            return False
        



class CustomScaler:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, arr):
        # Find the minimum and maximum values in the array
        self.min_val = min(arr)
        self.max_val = max(arr)

    def transform(self, data_point):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted. Call fit() before transform.")

        # Check if the range is zero to avoid division by zero
        if self.min_val == self.max_val:
            return 0.5  # Return 0.5 for a single point when range is zero, placing it in the middle of [0, 1]

        # Scale the data point to the range [0, 1]
        scaled_value = (data_point - self.min_val) / (self.max_val - self.min_val)
        
        return scaled_value
    


def generate_subsets(nums):
    def backtrack(start, current_subset):
        # Add the current subset to the list of subsets
        subsets.append(current_subset[:])
        
        # Explore all possible options to form subsets
        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()

    subsets = []
    backtrack(0, [])
    return subsets


def weighted_l1_norm(x1, x2):
    s = 0
    n = len(x2)
    for i in range(n):
        s += abs(x1[i] - x2[i])
    return s/n
