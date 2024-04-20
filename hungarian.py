import numpy as np

def hungarian(cost_matrix):
    cost_matrix = np.array(cost_matrix)
    
    n, m = cost_matrix.shape
    if m > n:
        cost_matrix = np.transpose(cost_matrix)
        n, m = m, n
        transposed = True
    
    if n > m:
        cost_matrix = np.pad(cost_matrix, [(0, 0), (0, n-m)], mode='constant', constant_values=np.max(cost_matrix))
        m = n

    # Step 1: Subtract the row minimum from each row
    cost_matrix -= np.min(cost_matrix, axis=1)[:, np.newaxis]

    # Step 2: Subtract the column minimum from each column
    cost_matrix -= np.min(cost_matrix, axis=0)

    print(cost_matrix)

    # Step 3: Find the minimum number of lines to cover all zeros in the cost matrix
    covered_rows = np.zeros(n, dtype=bool)
    covered_cols = np.zeros(m, dtype=bool)
    num_covered = 0
    while num_covered < n:
        # Find the minimum uncovered element
        minval = np.min(cost_matrix[~covered_rows, :][:, ~covered_cols])
        if np.isinf(minval):
            break

        # Subtract it from all uncovered elements
        cost_matrix[~covered_rows, :][:, ~covered_cols] -= minval

        print(cost_matrix)

        # Find the rows and columns that have a zero
        row_has_zero = np.any(cost_matrix[~covered_rows, :][:, ~covered_cols] == 0, axis=1)
        col_has_zero = np.any(cost_matrix[~covered_rows, :][:, ~covered_cols] == 0, axis=0)

        print(row_has_zero, col_has_zero)

        # Cover the rows and columns
        covered_rows |= row_has_zero
        covered_cols |= col_has_zero
        num_covered = np.sum(covered_rows)

        print(covered_rows, covered_cols)
        print(num_covered)

    # Step 4: Find a zero and star it. If there is no starred zero in its row or column, star it and
    # go to the next step. Repeat for each zero
    starred = np.zeros_like(cost_matrix, dtype=bool)
    primed = np.zeros_like(cost_matrix, dtype=bool)
    for i in range(n):
        for j in range(m):
            if cost_matrix[i, j] == 0 and not starred[i, :].any() and not starred[:, j].any():
                starred[i, j] = True
    
    # Step 5: Cover each column with a starred zero. If all columns are covered, the starred zeros describe
    # a complete set of unique assignments
    covered_cols = np.any(starred, axis=0)
    while not np.all(covered_cols):
        # Find an uncovered zero
        row, col = np.unravel_index(np.argmax(cost_matrix * ~starred), cost_matrix.shape)
        if starred[row, col]:
            # Find the starred zero in the row
            col = np.argmax(starred[row, :])
            primed[row, col] = True
            starred[row, col] = False
        else:
            starred[row, col] = True

        # Find the columns with a starred zero
        covered_cols = np.any(starred, axis=0)

    # Step 6: Construct a series of alternating primed and starred zeros as follows. Let Z0 represent the uncovered
    # primed zero found in Step 4. Let Z1 denote the starred zero in the column of Z0 (if any). Let Z2 denote the primed
    # zero in the row of Z1 (there will always be one). Continue until the series terminates at a primed zero that has no
    # starred zero in its column. Unstar each starred zero of the series, star each primed zero of the series, erase all
    # primes and uncover every line in the matrix. Return to Step 3
    path = np.zeros_like(cost_matrix, dtype=bool)
    path[primed] = True
    path[starred] = True
    while True:
        # Find the first primed zero in the path
        row, col = np.unravel_index(np.argmax(path * primed), path.shape)
        if not path[row, :].any():
            break

        # Find the starred zero in the same column
        row = np.argmax(starred[:, col])
        path[row, col] = True

        # Find the primed zero in the same row
        col = np.argmax(primed[row, :])
        path[row, col] = False
    
    # Step 7: Update the cost matrix
    # Find the minimum value of the cost matrix
    minval = np.min(cost_matrix[~path])
    # Add minval to the primed zeros
    cost_matrix[primed] += minval
    # Subtract minval from the starred zeros
    cost_matrix[starred] -= minval

    # Repeat until there are no uncovered zeros
    while not np.all(path):
        # Find the first uncovered zero
        row, col = np.unravel_index(np.argmax(~path * ~covered_rows[:, np.newaxis] * ~covered_cols, path.shape))
        minval = cost_matrix[row, col]

        # Subtract minval from every element in the same row
        cost_matrix[row, :] -= minval
        # Add minval to every element in the same column
        cost_matrix[:, col] += minval

        # Add the zero to the path
        path[row, col] = True

    # Step 8: Find the assignment
    assignment = np.zeros(n, dtype=int)
    for i in range(n):
        assignment[i] = np.argmax(path[i, :])

    return assignment

if __name__ == '__main__':
    # Example usage
    cost_matrix = np.random.randint(0, 10, (4, 4))
    hungarian(cost_matrix)