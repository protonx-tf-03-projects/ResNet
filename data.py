class Dataset:
    def __init__(self):
        pass

    def load_dataset(self):
	import os
        from argparse import ArgumentParser
        import tensorflow as tf
	import logging
	logging.basicConfig(level=logging.DEBUG)

	parser = ArgumentParser()
 	parser.add_argument("--logdir", default="logs")	
	home_dir = os.getcwd()
	
	parser.add_argument("--train-folder", default='{}/data/train'.format(home_dir), type=str)
	parser.add_argument("--valid-folder", default='{}/data/validation'.format(home_dir), type=str)
        parser.add_argument("--num-classes", default=2, type=int)
        parser.add_argument("--batch-size", default=32, type=int)
        parser.add_argument("--image-size", default=150, type=int)
        parser.add_argument("--patch-size", default=5, type=int)
        parser.add_argument("--validation-split", default=0.2, type=float)
        parser.add_argument("--image-channels", default=3, type=int)
        args = parser.parse_args()
  
        for i, arg in enumerate(vars(args)):
            print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
  
        train_folder = args.train_folder
        valid_folder = args.valid_folder
    
        # Load train images from folder
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_folder,
            subset="training",
            seed=123,
            image_size=(args.image_size, args.image_size),
            shuffle=True,
            validation_split = args.validation_split,
            batch_size=args.batch_size,)
       # Load Validation images from folder
       val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            valid_folder,
            subset="validation",
            seed=123,
            image_size=(args.image_size, args.image_size),
            shuffle=True,
            validation_split = args.validation_split,
            batch_size= args.batch_size,)

       assert args.image_size * args.image_size % ( args.patch_size * args.patch_size) == 0, 'Make sure that image-size is divisible by patch-size'
       assert args.image_channels == 3, 'Unfortunately, model accepts jpg images with 3 channels so far'

    def build_dataset(self):
        pass
