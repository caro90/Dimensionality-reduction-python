def generate_min_samples_range(dimensionality_size):

    starting_value = 3 * dimensionality_size
    rangeOfValues = []

    rangeBoolean = True
    while(rangeBoolean):
        # Define a range based on the size of the dataset
        if starting_value <= 2:
            rangeOfValues.extend([2])
            rangeBoolean = False
        else:
            rangeOfValues.append(starting_value)
        starting_value = round(starting_value/2)
    return rangeOfValues

# # Example usage
# dataset_sizes = [5000]  # Replace with actual dataset sizes
# for size in dataset_sizes:
#     min_samples_range = generate_min_samples_range(size)
#     print(f"Dataset size: {size}, min_samples range: {min_samples_range}")
