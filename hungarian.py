import numpy as np
import scipy.optimize as opt

def findMinimumLines (cost_matrix):
    rows, cols = cost_matrix.shape
    
    assigned_rows = np.zeros(rows, dtype=bool)
    assigned_cols = np.zeros(cols, dtype=bool)
    assignment_matrix = np.zeros(cost_matrix.shape, dtype=bool)

    zero_count_rows = np.sum(cost_matrix == 0, axis=1)
    zero_count_cols = np.sum(cost_matrix == 0, axis=0)
    zero_matrix = np.array(cost_matrix == 0, dtype=bool)

    # Assign columns in rows with only one zero (and vice versa), cross out the zeros in the corresponding assignments
    while True:
        assigned_row_or_col = False

        for row in range(rows):
            if zero_count_rows[row] != 1 or assigned_rows[row]:
                continue
            assigned_row_or_col = True
            assigned_col = np.argmax(zero_matrix[row, :])

            assignment_matrix[row, assigned_col] = True
            assigned_rows[row] = True
            assigned_cols[assigned_col] = True

            for r in range(rows):
                if zero_matrix[r, assigned_col]:
                    zero_matrix[r, assigned_col] = False
                    zero_count_rows[r] -= 1
                    zero_count_cols[assigned_col] -= 1

        for col in range(cols):
            if zero_count_cols[col] != 1 or assigned_cols[col]:
                continue
            assigned_row_or_col = True
            assigned_row = np.argmax(zero_matrix[:, col])

            assignment_matrix[assigned_row, col] = True
            assigned_rows[assigned_row] = True
            assigned_cols[col] = True

            for c in range(cols):
                if zero_matrix[assigned_row, c]:
                    zero_matrix[assigned_row, c] = False
                    zero_count_rows[assigned_row] -= 1
                    zero_count_cols[c] -= 1

        if not assigned_row_or_col:
            break

    # If needed, arbitrary assignment of zeros
    if np.max(zero_count_rows) != 0 or np.max(zero_count_cols) != 0:
        for row in range(rows):
            if zero_count_rows[row] == 0 or assigned_rows[row]:
                continue
            for col in range(cols):
                if zero_count_cols[col] == 0 or assigned_cols[col]:
                    continue
                if zero_matrix[row, col]:
                    assignment_matrix[row, col] = True
                    assigned_rows[row] = True
                    assigned_cols[col] = True
                    for r in range(rows):
                        if zero_matrix[r, col]:
                            zero_matrix[r, col] = False
                            zero_count_rows[r] -= 1
                            zero_count_cols[col] -= 1

                    for c in range(cols):
                        if zero_matrix[row, c]:
                            zero_matrix[row, c] = False
                            zero_count_rows[row] -= 1
                            zero_count_cols[c] -= 1

    ticked_rows = np.zeros(rows, dtype=bool)
    ticked_cols = np.zeros(cols, dtype=bool)
    zero_matrix = np.array(cost_matrix == 0, dtype=bool)

    # Tick all unassigned rows
    ticked_rows = ~assigned_rows

    while True:
        ticked_row_or_col = False

        # In all ticked rows, tick all columns corresponding with zeros in those rows
        for row in range(rows):
            if not ticked_rows[row]:
                continue
            for col in range(cols):
                if zero_matrix[row, col] and not ticked_cols[col]:
                    ticked_cols[col] = True
                    ticked_row_or_col = True

        # In all ticked columns, tick all rows corresponding with assignments in those columns
        for col in range(cols):
            if not ticked_cols[col]:
                continue
            for row in range(rows):
                if assignment_matrix[row, col] and not ticked_rows[row]:
                    ticked_rows[row] = True
                    ticked_row_or_col = True

        if not ticked_row_or_col:
            break

    # Cover all unticked rows and ticked columns
    covered_rows = ~ticked_rows
    covered_cols = ticked_cols
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

    print("Step 3: Find minimum lines\n", cost_matrix, num_covered)


    # Step 4: Assign rows to columns to minimize the cost
    col_ind_assignments = np.zeros(N, dtype=int)
    row_ind_assignments = np.zeros(N, dtype=int)
    assigned_rows = np.zeros(N, dtype=bool)
    assigned_cols = np.zeros(N, dtype=bool)
    num_assigned = 0

    ##Debugging
    # two_count = 0
    while True:
        # Initialize zero counts of rows and columns
        # If a row or column is assigned, set the zero count to N + 1
        zero_count_rows = np.zeros(N, dtype=int)
        zero_count_cols = np.zeros(N, dtype=int)
        for row in range(N):
            for col in range(N):
                if cost_matrix[row, col] == 0 and not assigned_rows[row] and not assigned_cols[col]:
                    zero_count_rows[row] += 1
                    zero_count_cols[col] += 1

        zero_count_matrix = np.zeros((N, N), dtype=int)
        for row in range(N):
            for col in range(N):
                if cost_matrix[row, col] != 0 or assigned_rows[row] or assigned_cols[col]:
                    zero_count_matrix[row, col] = 2 * N + 1
                    continue
                zero_count_matrix[row, col] = min(zero_count_rows[row], zero_count_cols[col])
                

        # Find the row and column with the minimum number of zeros and assign the zero
        min_index = np.argmin(zero_count_matrix)
        min_index = (min_index // N, min_index % N)
        col_ind_assignments[min_index[0]] = min_index[1]
        row_ind_assignments[min_index[1]] = min_index[0]
        assigned_rows[min_index[0]] = True
        assigned_cols[min_index[1]] = True
        num_assigned += 1

        ##Debugging
        # print("Zero count matrix:\n", zero_count_matrix)
        # print("assigned:", min_index, zero_count_matrix[min_index[0], min_index[1]])
        # if zero_count_matrix[min_index[0], min_index[1]] == 2:
        #     two_count += 1

        if num_assigned == N:
            ## Debugging
            # if np.sum(col_ind_assignments == 0) > 1 or np.sum(row_ind_assignments == 0) > 1:
            #     print("Cost matrix:\n", cost_matrix)
            #     print("Zero count matrix:\n", zero_count_matrix)
            #     print("two_count:", two_count)
            #     print("Row and Col index assignments\n", row_ind_assignments, col_ind_assignments)
            #     print("exited")
            #     exit()
            break

    
    print("Step 4: Assign rows to columns\n", cost_matrix)
    print("Row and Col index assignments\n", row_ind_assignments, col_ind_assignments)


    # return assignments of length max(objects, predictions)
    # if #objects > #predictions, then assignments[object index] > #predictions means it is a new object
    # if #predictions > #objects, then object index > #objects means it is a lost object
    return col_ind_assignments #, row_ind_assignments



if __name__ == '__main__':
    ## Debugging
    cost_matrix = np.random.randint(0, 100, (100, 100))
    # cost_matrix = np.array([[11, 7, 10, 17, 10], [13, 21, 7, 11, 13], [13, 13, 15, 13, 14], [18, 10, 13, 16, 14], [12, 8, 16, 19, 10]])
    hungarian(cost_matrix)
    # for _ in range(1000):
    #     print("Iteration:", _)
    #     cost_matrix = np.random.randint(0, 100, (20, 20))
    #     hungarian(cost_matrix)