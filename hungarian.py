import numpy as np
import scipy.optimize as opt

def findMinimumLines (cost_matrix):
    rows, cols = cost_matrix.shape
    
    # Arbitrarily assign zeros in the cost matrix to rows and columns
    assigned_rows = np.zeros(rows, dtype=bool)
    assigned_cols = np.zeros(cols, dtype=bool)
    assignment_matrix = np.zeros(cost_matrix.shape, dtype=bool)

    for row in range(rows):
        for col in range(cols):
            if cost_matrix[row, col] == 0 and not assigned_rows[row] and not assigned_cols[col]:
                assignment_matrix[row, col] = True
                assigned_rows[row] = True
                assigned_cols[col] = True

    # Increase the number of assigned zeros incrementally until all zeros are covered
    covered_rows = np.zeros(rows, dtype=bool)
    covered_cols = np.copy(assigned_cols)

    while True:
        # Cover all columns with assigned zeros
        primed_matrix = np.zeros(cost_matrix.shape, dtype=bool)
        covered_rows = np.zeros(rows, dtype=bool)
        covered_cols = np.copy(assigned_cols)
        reset = False
        
        while True:
            # Search for uncovered zeros in the cost matrix
            prime_found = False
            for row in range(rows):
                if covered_rows[row]:
                    continue

                primed_col = -1
                for col in range(cols):
                    # Prime an uncovered zero
                    if cost_matrix[row, col] != 0 or covered_cols[col]:
                        continue
                    primed_matrix[row, col] = True
                    prime_found = True
                    primed_col = col
                    if assigned_rows[row]:
                        covered_rows[row] = True
                    break
                else:
                    continue

                # If the primed zero has an assigned zero in the same row, cover the row and uncover the column
                if covered_rows[row]:
                    for col in range(cols):
                        if assignment_matrix[row, col]:
                            covered_cols[col] = False
                            break
                # If the primed zero has no assigned zero in the same row, find an alternating path of primed and assigned zeros
                # Starting from the primed zero, search the column for an assigned zero, then search the assigned zero's row for a primed zero, etc.
                # When the path ends, convert the primed zeros to assigned zeros and assigned zeros to unassigned zeros
                # Then, reset the coverage and primed zeros and start the coverage process again
                else:
                    row_path = [row]
                    col_path = [primed_col]

                    while True:
                        for r in range(rows):
                            if assignment_matrix[r, col_path[-1]]:
                                row_path.append(r)
                                break

                        if len(row_path) == len(col_path):
                            break

                        for c in range(cols):
                            if primed_matrix[row_path[-1], c]:
                                col_path.append(c)
                                break

                    for col_path_index in range(len(col_path)):
                        row_path_index = col_path_index
                        if primed_matrix[row_path[row_path_index], col_path[col_path_index]]:
                            primed_matrix[row_path[row_path_index], col_path[col_path_index]] = False
                            assignment_matrix[row_path[row_path_index], col_path[col_path_index]] = True
                            assigned_rows[row_path[row_path_index]] = True
                            assigned_cols[col_path[col_path_index]] = True

                        if row_path_index + 1 == len(row_path):
                            break

                        if assignment_matrix[row_path[row_path_index + 1], col_path[col_path_index]]:
                            assignment_matrix[row_path[row_path_index + 1], col_path[col_path_index]] = False
                    
                    reset = True
                    break
            
            if not prime_found or reset:
                break
        if not prime_found:
            break

    num_covered = np.sum(covered_rows) + np.sum(covered_cols)
    return num_covered, covered_rows, covered_cols

def hungarian(cost_matrix, test=True):
    if not test:
        row_ind_assignments, col_ind_assignments = opt.linear_sum_assignment(cost_matrix)
        return col_ind_assignments
    
    # cost_matrix : rows are objects and columns are predictions
    cost_matrix = np.array(cost_matrix)
    num_rows, num_cols = cost_matrix.shape
    N = max(num_rows, num_cols)

    # Preprocessing: if rows > columns or columns > rows, pad the matrix with max value or 0 respectively
    if num_rows > num_cols:
        # cost_matrix = np.pad(cost_matrix, [(0, 0), (0, num_rows-num_cols)], mode='constant', constant_values=np.inf)
        cost_matrix = np.pad(cost_matrix, [(0, 0), (0, num_rows-num_cols)], mode='constant', constant_values=np.max(cost_matrix))
    elif num_cols > num_rows:
        cost_matrix = np.pad(cost_matrix, [(0, num_cols-num_rows), (0, 0)], mode='constant', constant_values=0)

    print("Preprocessing: Pad the cost matrix\n", cost_matrix)


    # Step 1: Subtract the row minimum from each row
    cost_matrix -= np.min(cost_matrix, axis=1)[:, np.newaxis]

    print("Step 1: Subtract row minimum\n", cost_matrix)


    # Step 2: Subtract the column minimum from each column
    cost_matrix -= np.min(cost_matrix, axis=0)

    print("Step 2: Subtract column minimum\n", cost_matrix)


    # Step 3: Find the minimum number of lines to cover all zeros in the cost matrix
    num_covered, covered_rows, covered_cols = findMinimumLines(cost_matrix)

    while num_covered < N:
        print("Step 3: Find minimum lines\n", cost_matrix, num_covered, covered_rows, covered_cols)
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
    col_ind_assignments = np.zeros(N, dtype=int)
    row_ind_assignments = np.zeros(N, dtype=int)
    assigned_rows = np.zeros(N, dtype=bool)
    assigned_cols = np.zeros(N, dtype=bool)
    num_assigned = 0

    while True:
        # Initialize zero counts of rows and columns
        # If a row or column is assigned, set the zero count to N + 1
        zero_count_rows = np.zeros(N, dtype=int)
        zero_count_cols = np.zeros(N, dtype=int)
        for n in range(N):
            if assigned_rows[n]:
                zero_count_rows[n] = N + 1
            if assigned_cols[n]:
                zero_count_cols[n] = N + 1
        for row in range(N):
            for col in range(N):
                if cost_matrix[row, col] == 0 and not assigned_rows[row] and not assigned_cols[col]:
                    zero_count_rows[row] += 1
                    zero_count_cols[col] += 1
        
        # Find the row or column with the minimum number of zeros and assign the first zero seen
        row_to_assign = True if np.min(zero_count_rows) <= np.min(zero_count_cols) else False
        min_index = np.argmin(zero_count_rows) if row_to_assign else np.argmin(zero_count_cols)
        for n in range(N):
            if row_to_assign:
                if cost_matrix[min_index, n] == 0 and not assigned_cols[n]:
                    col_ind_assignments[min_index] = n
                    row_ind_assignments[n] = min_index
                    assigned_rows[min_index] = True
                    assigned_cols[n] = True
                    num_assigned += 1
                    break
            else:
                if cost_matrix[n, min_index] == 0 and not assigned_rows[n]:
                    col_ind_assignments[n] = min_index
                    row_ind_assignments[min_index] = n
                    assigned_rows[n] = True
                    assigned_cols[min_index] = True
                    num_assigned += 1
                    break

        if num_assigned == N:
            break

    print("Step 4: Assign rows to columns\n", cost_matrix)
    print("Row and Col index assignments\n", row_ind_assignments, col_ind_assignments)


    # return assignments of length max(objects, predictions)
    # if #objects > #predictions, then assignments[object index] > #predictions means it is a new object
    # if #predictions > #objects, then object index > #objects means it is a lost object
    return col_ind_assignments #, row_ind_assignments



if __name__ == '__main__':
    # Example usage
    # cost_matrix = np.random.randint(0, 100, (10, 10))
    cost_matrix = np.array([[99, 4, 19, 68, 9], [62, 94, 84, 15, 93], [71, 64, 72, 82, 76], [71, 81, 79, 19, 33], [0, 0, 0, 0, 0]])
    hungarian(cost_matrix)