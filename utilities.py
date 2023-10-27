
def matrix_to_matlab(mat, name):

    print(name + ' = [', end='')
    for row in range(mat.shape[0]):
        if row > 0:
            print('\t', end='')
        for col in range(mat.shape[1]):
            print(mat[row][col], end=', ')
        
        if row < mat.shape[0] - 1:
            print(';')
    print('];')