import os
"""
    Input:
        n: int - the number of class
    Output:
        Folder structure:
            main_directory/
                .../data
                    .../train
                        ...class_a/
                                ......a_image_1.jpg
                                ......a_image_2.jpg
                        ...class_b/
                            ......b_image_1.jpg
                            ......b_image_2.jpg
                    .../validation
                        ...class_a/
                                ......a_image_1.jpg
                                ......a_image_2.jpg
                        ...class_b/
                            ......b_image_1.jpg
                            ......b_image_2.jpg
"""
try:
    number = int(input('Enter the number of classes: '))
    if(number>1):
        name = []
        for type in range(number):
            name.append(input("Type the name of class {}: ".format(type+1)))
        os.mkdir('data')
        os.chdir('data')
        path = ['train', 'validation']
        for i in path:
            os.mkdir(i)
            os.chdir('{}/'.format(i))
            for j in name:
                os.mkdir(j)
            os.chdir('../')
        print('Data including {} classes folder have been created!'.format(number))
    else:
        print('Please type the number of classes greater than 1!')
except ValueError:
    print('Please type number!')
except FileExistsError:
    print('Cannot create a file when that file already exists!')
    print("""Please remove "Data" folder!""")
