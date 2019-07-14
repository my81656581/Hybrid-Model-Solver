import numpy as np

def foo(desktop: dict):
    a = desktop['a']
    b = desktop['b']
    desktop['c'] = a + b

def display(desktop):
    print('=' * 16)
    for k, v in desktop.items():
        print(k, v)
    print('=' * 16 + '\n')

def main():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    desktop = {'a': a, 'b': b}
    display(desktop)
    a += 1
    display(desktop)
    desktop['b'] += 2
    display(desktop)
    foo(desktop)
    display(desktop)

if __name__ == '__main__':
    main()