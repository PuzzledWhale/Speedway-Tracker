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

    # print("Preprocessing: Pad the cost matrix\n", cost_matrix)

    # Step 1: Subtract the row minimum from each row
    cost_matrix -= np.min(cost_matrix, axis=1)[:, np.newaxis]

    # print("Step 1: Subtract row minimum\n", cost_matrix)

    # Step 2: Subtract the column minimum from each column
    cost_matrix -= np.min(cost_matrix, axis=0)

    # print("Step 2: Subtract column minimum\n", cost_matrix)

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

    # print("Step 3: Find minimum lines\n", cost_matrix, covered_rows, covered_cols)
    
    # Step 4: Assign rows to columns to minimize the cost
    assignments = np.zeros(rows, dtype=int)
    zero_count_rows = np.sum(cost_matrix == 0, axis=1)
    zero_count_cols = np.sum(cost_matrix == 0, axis=0)
    
    assigned_rows = np.zeros(rows, dtype=bool)
    assigned_cols = np.zeros(cols, dtype=bool)
    num_assigned = 0
    prev_num_assigned = 0

    while num_assigned < rows:
        prev_num_assigned = num_assigned
        for i in range(rows):
            if zero_count_rows[i] == 1 and not assigned_rows[i]:
                for j in range(cols):
                    if cost_matrix[i, j] == 0 and not assigned_cols[j]:
                        assignments[i] = j
                        assigned_rows[i] = True
                        assigned_cols[j] = True
                        num_assigned += 1
                        break
                break
        if num_assigned == prev_num_assigned:
            for i in range(cols):
                if zero_count_cols[i] == 1 and not assigned_cols[i]:
                    for j in range(rows):
                        if cost_matrix[j, i] == 0 and not assigned_rows[j]:
                            assignments[j] = i
                            assigned_rows[j] = True 
                            assigned_cols[i] = True
                            num_assigned += 1
                            break
                    break
        if num_assigned < rows:
            zero_count_rows = np.zeros(rows, dtype=int)
            zero_count_cols = np.zeros(cols, dtype=int)
            for i in range(rows):
                for j in range(cols):
                    if cost_matrix[i, j] == 0 and not assigned_rows[i] and not assigned_cols[j]:
                        zero_count_rows[i] += 1
                        zero_count_cols[j] += 1
    #     print("Cost Matrix:\n", cost_matrix)
    #     print("Zero count rows:", zero_count_rows)
    #     print("Zero count cols:", zero_count_cols)
    #     print("Assigned rows:", assigned_rows)
    #     print("Assigned cols:", assigned_cols)

                

    # print("Step 4: Assign rows to columns\n", cost_matrix, assignments)

    # return assignments of length max(objects, predictions)
    # if #objects > #predictions, then assignments[object index] > #predictions means it is a new object
    # if #predictions > #objects, then object index > #objects means it is a lost object
    return assignments

def findMinimumLines (cost_matrix):
    rows, cols = cost_matrix.shape
    
    # Step 3.1: Find the minimum number of lines to cover all zeros in the cost matrix with row major priority
    zero_count_rows = np.zeros(rows, dtype=int)
    zero_count_cols = np.zeros(cols, dtype=int)

    for i in range(rows):
        for j in range(cols):
            if cost_matrix[i, j] == 0:
                zero_count_rows[i] += 1
                zero_count_cols[j] += 1

    zero_count_rows_max = np.max(zero_count_rows)
    zero_count_cols_max = np.max(zero_count_cols)

    rmajor_covered_rows = np.zeros(rows, dtype=bool)
    rmajor_covered_cols = np.zeros(cols, dtype=bool)
    rmajor_num_covered = 0

    while zero_count_rows_max > 0 or zero_count_cols_max > 0:
        if zero_count_rows_max >= zero_count_cols_max:
            # Cover the row with the most zeros
            row = np.argmax(zero_count_rows)
            rmajor_covered_rows[row] = True
            zero_count_rows[row] = 0
            for j in range(cols):
                if cost_matrix[row, j] == 0:
                    zero_count_cols[j] -= 1
        else:
            # Cover the column with the most zeros
            col = np.argmax(zero_count_cols)
            rmajor_covered_cols[col] = True
            zero_count_cols[col] = 0
            for i in range(rows):
                if cost_matrix[i, col] == 0:
                    zero_count_rows[i] -= 1

        rmajor_num_covered += 1
        zero_count_rows_max = np.max(zero_count_rows)
        zero_count_cols_max = np.max(zero_count_cols)

    # Step 3.2: Find the minimum number of lines to cover all zeros in the cost matrix with column major priority
    zero_count_rows = np.zeros(rows, dtype=int)
    zero_count_cols = np.zeros(cols, dtype=int)

    for i in range(rows):
        for j in range(cols):
            if cost_matrix[i, j] == 0:
                zero_count_rows[i] += 1
                zero_count_cols[j] += 1

    zero_count_rows_max = np.max(zero_count_rows)
    zero_count_cols_max = np.max(zero_count_cols)
    
    cmajor_covered_rows = np.zeros(rows, dtype=bool)
    cmajor_covered_cols = np.zeros(cols, dtype=bool)
    cmajor_num_covered = 0

    while zero_count_rows_max > 0 or zero_count_cols_max > 0:
        if zero_count_cols_max >= zero_count_rows_max:
            # Cover the column with the most zeros
            col = np.argmax(zero_count_cols)
            cmajor_covered_cols[col] = True
            zero_count_cols[col] = 0
            for i in range(rows):
                if cost_matrix[i, col] == 0:
                    zero_count_rows[i] -= 1
        else:
            # Cover the row with the most zeros
            row = np.argmax(zero_count_rows)
            cmajor_covered_rows[row] = True
            zero_count_rows[row] = 0
            for j in range(cols):
                if cost_matrix[row, j] == 0:
                    zero_count_cols[j] -= 1

        cmajor_num_covered += 1
        zero_count_rows_max = np.max(zero_count_rows)
        zero_count_cols_max = np.max(zero_count_cols)

    if rmajor_num_covered <= cmajor_num_covered:
        return rmajor_num_covered, rmajor_covered_rows, rmajor_covered_cols
    return cmajor_num_covered, cmajor_covered_rows, cmajor_covered_cols

if __name__ == '__main__':
    # Example usage
    cost_matrix = np.random.randint(0, 10, (5, 5))
    hungarian(cost_matrix)