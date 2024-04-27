import numpy as np

def hungarian(cost_matrix):
    # cost_matrix : rows are objects and columns are predictions
    cost_matrix = np.array(cost_matrix)
    
    rows, cols = cost_matrix.shape
    # if rows > columns or columns > rows, pad the matrix with max value
    if rows > cols:
        cost_matrix = np.pad(cost_matrix, [(0, 0), (0, rows-cols)], mode='constant', constant_values=np.max(cost_matrix))
        cols = rows
    elif cols > rows:
        cost_matrix = np.pad(cost_matrix, [(0, cols-rows), (0, 0)], mode='constant', constant_values=np.max(cost_matrix))
        rows = cols

    print("Preprocessing: Pad the cost matrix\n", cost_matrix)

    # Step 1: Subtract the row minimum from each row
    cost_matrix -= np.min(cost_matrix, axis=1)[:, np.newaxis]

    print("Step 1: Subtract row minimum\n", cost_matrix)

    # Step 2: Subtract the column minimum from each column
    cost_matrix -= np.min(cost_matrix, axis=0)

    print("Step 2: Subtract column minimum\n", cost_matrix)

    # Step 3: Find the minimum number of lines to cover all zeros in the cost matrix
    num_covered, covered_rows, covered_cols = findMinimumLines(cost_matrix)

    while num_covered < rows:
        # Find the minimum uncovered element
        minval = np.min(cost_matrix[~covered_rows, :][:, ~covered_cols])
        
        # Subtract minval from every element in the uncovered rows
        cost_matrix[~covered_rows, :] -= minval

        # Add minval to every element in the covered columns
        cost_matrix[:, covered_cols] += minval

        # Find the minimum number of lines to cover all zeros in the cost matrix
        num_covered, covered_rows, covered_cols = findMinimumLines(cost_matrix)

    print("Step 3: Find minimum lines\n", cost_matrix, covered_rows, covered_cols)
    
    # Step 4: Assign rows to columns to minimize the cost
    assignments = np.zeros(rows, dtype=int)
    zero_count_rows = np.sum(cost_matrix == 0, axis=1)
    
    assigned_rows = np.zeros(rows, dtype=bool)
    assigned_cols = np.zeros(cols, dtype=bool)
    num_assigned = 0

    while num_assigned < rows:
        for i in range(rows):
            if zero_count_rows[i] == 1:
                for j in range(cols):
                    if cost_matrix[i, j] == 0 and not assigned_rows[i] and not assigned_cols[j]:
                        assignments[i] = j
                        assigned_rows[i] = True
                        assigned_cols[j] = True
                        num_assigned += 1
                        break
                break
        if num_assigned < rows:
            zero_count_rows = np.zeros(rows, dtype=int)
            for i in range(rows):
                for j in range(cols):
                    if cost_matrix[i, j] == 0 and not assigned_rows[i] and not assigned_cols[j]:
                        zero_count_rows[i] += 1
                

    print("Step 4: Assign rows to columns\n", cost_matrix, assignments)

    # return assignments of length max(objects, predictions)
    # if #objects > #predictions, then assignments[object index] > #predictions means it is a new object
    # if #predictions > #objects, then object index > #objects means it is a lost object
    return assignments

def findMinimumLines (cost_matrix):
    rows, cols = cost_matrix.shape
    zero_count_rows = np.zeros(rows, dtype=int)
    zero_count_cols = np.zeros(cols, dtype=int)

    # Step 3: Find the minimum number of lines to cover all zeros in the cost matrix
    for i in range(rows):
        for j in range(cols):
            if cost_matrix[i, j] == 0:
                zero_count_rows[i] += 1
                zero_count_cols[j] += 1

    zero_count_rows_max = np.max(zero_count_rows)
    row = np.argmax(zero_count_rows)
    zero_count_cols_max = np.max(zero_count_cols)
    col = np.argmax(zero_count_cols)

    covered_rows = np.zeros(rows, dtype=bool)
    covered_cols = np.zeros(cols, dtype=bool)
    num_covered = 0

    while zero_count_rows_max > 0 or zero_count_cols_max > 0:
        if zero_count_rows_max >= zero_count_cols_max:
            # Cover the row with the most zeros
            row = np.argmax(zero_count_rows)
            covered_rows[row] = True
            zero_count_rows[row] = 0
            for j in range(cols):
                if cost_matrix[row, j] == 0:
                    zero_count_cols[j] -= 1
        else:
            # Cover the column with the most zeros
            col = np.argmax(zero_count_cols)
            covered_cols[col] = True
            zero_count_cols[col] = 0
            for i in range(rows):
                if cost_matrix[i, col] == 0:
                    zero_count_rows[i] -= 1

        num_covered += 1
        zero_count_rows_max = np.max(zero_count_rows)
        row = np.argmax(zero_count_rows)
        zero_count_cols_max = np.max(zero_count_cols)
        col = np.argmax(zero_count_cols)

    return num_covered, covered_rows, covered_cols

if __name__ == '__main__':
    # Example usage
    cost_matrix = np.random.randint(0, 10, (4, 4))
    hungarian(cost_matrix)