import os, shutil
#from scipy.misc import imsave
import cv2
from sklearn.model_selection import KFold, RepeatedKFold
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
from keras.optimizers import RMSprop, SGD, Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, save_img, image
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from segmentation_models import Unet


###################################################
####################Loading Data###################
###################################################
def A_List_of_Data_Path():
    #######################A List of Images Path#####################
    images_path_list = [os.path.join(image_path, image) for image in os.listdir(image_path)]
    print("list of images:", os.listdir(image_path))

    ###########Checking dimension, shape, type, format, mode, and size of Images##########
    img_dim_list = []
    img_shape_list = []
    img_type_list = []
    img_format_list = []
    img_mode_list = []
    img_size_list = []
    for img_dir in images_path_list:
        print("img_dir:", img_dir)
        img_prop = cv2.imread(img_dir)
        converted_img = cv2.cvtColor(img_prop, cv2.COLOR_RGBA2RGB)
        converted_img = cv2.resize(converted_img, (224, 224))
        converted_img = array_to_img(converted_img, data_format='channels_last', dtype='float32')
        converted_img.save(img_dir)

    for img_dir in images_path_list:

         img = Image.open(img_dir)
         image_prop = plt.imread(img_dir)
         img_dim_list.append(image_prop.ndim)
         img_shape_list.append(image_prop.shape)
         img_type_list.append(image_prop.dtype)
         img_format_list.append(img.format)
         img_mode_list.append(img.mode)
         img_size_list.append(img.size)

    print("dimension of image:", img_dim_list)
    print("shape of image:", img_shape_list)
    print("type of image:", img_type_list)
    print("format of image:", img_format_list)
    print("new mode of image:", img_mode_list)
    print("size of image:", img_size_list)

    #Showing an image
    img_dir = images_path_list[3]
    img = image.load_img(img_dir)
    plt.imshow(img)
    plt.show()

    #####################################################################################
    #######################A List of Mask Path###########################################
    masks_path_list = [os.path.join(mask_path, mask) for mask in os.listdir(mask_path)]
    #print("names of masks:", os.listdir(mask_path))
    # print("masks path list:", masks_path_list)
    ###########Checkinng dimension, shape, type, format, mode, and size of Masks#########
    mask_dim_list = []
    mask_shape_list = []
    mask_type_list = []
    mask_format_list = []
    mask_mode_list = []
    mask_size_list = []
    for mask_dir in masks_path_list:
        mask = Image.open(mask_dir)
        mask = mask.resize((224, 224))
        mask.save(mask_dir)
    for mask_dir in masks_path_list:
        mask_prop = cv2.imread(mask_dir)
        mask = Image.open(mask_dir)
        mask_dim_list.append(mask_prop.ndim)
        mask_shape_list.append(mask_prop.shape)
        mask_type_list.append(mask_prop.dtype)
        mask_format_list.append(mask.format)
        mask_mode_list.append(mask.mode)
        mask_size_list.append(mask.size)

    print("dimension of mask:", mask_dim_list)
    print("shape of mask:", mask_shape_list)
    print("type of mask:", mask_type_list)
    print("format of mask:", mask_format_list)
    print("new mode of mask:", mask_mode_list)
    print("size of mask:", mask_size_list)

    #Showing a mask
    mask_dir = masks_path_list[3]
    mask = image.load_img(mask_dir)
    plt.imshow(mask)
    plt.show()

    images_masks_path_list = [images_path_list, masks_path_list]
    return images_masks_path_list

###################################################
##################Designing the Model###################
###################################################
def build_model():

    model = Unet(backbone_name='mobilenetv2',
                 input_shape= (224, 224, 3),
                 classes=1,
                 activation='sigmoid',
                 encoder_weights= weight_mobilenetv2_path,
                 encoder_freeze= True,
                 encoder_features='default',
                 decoder_block_type='upsampling',
                 decoder_filters= (256, 128, 64, 32, 16),
                 decoder_use_batchnorm=True)

    #model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['acc'])
    #model.compile(loss='binary_crossentropy', optimizer=SGD(lr=1e-4, momentum=0.9), metrics=['acc'])
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['acc'])
    model.summary()
    return model

##################################################
###########Training and Evaluating Model##########
##################################################
def train_and_evaluate_model(images_masks_path_list, model):
    images_path_list = images_masks_path_list[0]
    masks_path_list = images_masks_path_list[1]

    #######Defining K-fold cross validation###########
    K = 5
    kfold = KFold(n_splits=K, shuffle=False, random_state=None)
    ##############Saving the validation logs at each fold############
    all_accu_training = []
    all_accu_validation = []
    all_loss_training = []
    all_loss_validation = []
    for train_index, validation_index in kfold.split(images_path_list, masks_path_list):

        train_images = [images_path_list[i] for i in train_index]
        train_masks = [masks_path_list[i] for i in train_index]
        validation_images = [images_path_list[i] for i in validation_index]
        validation_masks = [masks_path_list[i] for i in validation_index]

        ########################Directories########################
        segment_path = os.path.join(base_path, 'Semantic_Segmentation')
        os.makedirs(segment_path, exist_ok=True)
        #####Directories for the training and validation splits####
        train_path = os.path.join(segment_path, 'train')
        os.makedirs(train_path, exist_ok=True)

        validation_path = os.path.join(segment_path, 'validation')
        os.makedirs(validation_path, exist_ok=True)
        #############Directory with training images###############
        train_images_path = os.path.join(train_path, 'images')
        os.makedirs(train_images_path, exist_ok=True)
        train_images_subfolder_path = os.path.join(train_images_path, 'subfolder')
        os.makedirs(train_images_subfolder_path, exist_ok=True)
        #############Directory with training masks###############
        train_masks_path = os.path.join(train_path, 'masks')
        os.makedirs(train_masks_path, exist_ok=True)
        train_masks_subfolder_path = os.path.join(train_masks_path, 'subfolder')
        os.makedirs(train_masks_subfolder_path, exist_ok=True)

        #############Directory with validation images###############
        validation_images_path = os.path.join(validation_path, 'images')
        os.makedirs(validation_images_path, exist_ok=True)
        validation_images_subfolder_path = os.path.join(validation_images_path, 'subfolder')
        os.makedirs(validation_images_subfolder_path, exist_ok=True)
        #############Directory with validation masks###############
        validation_masks_path = os.path.join(validation_path, 'masks')
        os.makedirs(validation_masks_path, exist_ok=True)
        validation_masks_subfolder_path = os.path.join(validation_masks_path, 'subfolder')
        os.makedirs(validation_masks_subfolder_path, exist_ok=True)
        #####Copy of images in train_images to train_images_path####
        for image in train_images:
            image_name = image.split("/")[-1]
            # rc: right_copy
            rc = os.path.join(image_path, image_name)
            # fp: forge_past
            fp = os.path.join(train_images_subfolder_path, image_name)
            shutil.copyfile(rc, fp)
        #####Copy of masks in train_masks to train_masks_path####
        for mask in train_masks:
            mask_name = mask.split("/")[-1]
            rc = os.path.join(mask_path, mask_name)
            fp = os.path.join(train_masks_subfolder_path, mask_name)
            shutil.copyfile(rc, fp)
        #####Copy of images in validation_images to validation_images_path####
        for image in validation_images:
            image_name = image.split("/")[-1]
            rc = os.path.join(image_path, image_name)
            fp = os.path.join(validation_images_subfolder_path, image_name)
            shutil.copyfile(rc, fp)
        #####Copy of masks in validation_masks to validation_masks_path####
        for mask in validation_masks:
            mask_name = mask.split("/")[-1]
            rc = os.path.join(mask_path, mask_name)
            fp = os.path.join(validation_masks_subfolder_path, mask_name)
            shutil.copyfile(rc, fp)

        #####Augmentation of Training images and masks####
        # we create two instances with the same arguments

        data_gen_args = dict(rescale=1. / 255,
                             rotation_range=40,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

        train_images_datagen = ImageDataGenerator(**data_gen_args)
        train_masks_datagen = ImageDataGenerator(**data_gen_args)

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        train_images_generator = train_images_datagen.flow_from_directory(train_images_path, target_size=(224, 224),
                                                                          color_mode="rgb",
                                                                          batch_size=5, class_mode=None, seed=seed)
        train_masks_generator = train_masks_datagen.flow_from_directory(train_masks_path, target_size=(224, 224),
                                                                        color_mode="grayscale",
                                                                        batch_size=5, class_mode=None, seed=seed)

        # Combine generators into one which yields image and masks
        train_generator = zip(train_images_generator, train_masks_generator)

        #Checking_data_augmentation
        images , masks = None, None
        for i, m in train_generator:
            images, masks = i, m
            break

        plt.imshow(images[3])
        plt.imshow(np.reshape(masks[3], (224,224)))
        plt.show()

        ############Rescaling of Validation data################
        validation_images_datagen = ImageDataGenerator(rescale=1. / 255, )
        validation_masks_datagen = ImageDataGenerator(rescale=1. / 255)
        ### Validation_images_masks
        validation_images_generator = validation_images_datagen.flow_from_directory(validation_images_path, target_size=(224, 224),
                                                                                    batch_size=5, color_mode="rgb", class_mode=None)
        validation_masks_generator = validation_masks_datagen.flow_from_directory(validation_masks_path, target_size=(224, 224),
                                                                                  batch_size=5, color_mode="grayscale",
                                                                                  class_mode=None)
        # Combine generators into one which yields image and masks
        validation_generator = zip(validation_images_generator, validation_masks_generator)
        ###############Training the Model###################################################
        # saves the model weights after each epoch if the validation loss decreased
        model_weights_path = os.path.join(path_to_save_model, "weights_s50_e30.hdf5")
        model_weights = h5py.File(model_weights_path, 'w')
        model_weights.close()

        earlystopper = EarlyStopping(patience=5, verbose=1)
        checkpointer = ModelCheckpoint(model_weights_path, verbose=1, save_best_only=True)

        history = model.fit_generator(train_generator, steps_per_epoch= 50 , epochs= 30, callbacks=[earlystopper, checkpointer],
                                      validation_data=validation_generator,
                                      validation_steps=1)
        history_dict = history.history
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        accu = history_dict['acc']
        val_accu = history_dict['val_acc']
        print("accu_history:", accu)
        all_accu_training.append(accu)
        all_accu_validation.append(val_accu)
        all_loss_training.append(loss)
        all_loss_validation.append(val_loss)

        ###############################################################################
        ############Displaying curves of loss and accuracy during training#############
        ###############################################################################

        epochs = range(1, len(accu) + 1)
        plt.plot(epochs, accu, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accu, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.ylim([0, 2])
        plt.legend()
        plt.show()


        # Final Work: Deleting Semantic_Segmentation Folder of base_path
        ################Final Work########################
        shutil.rmtree(segment_path)

        return model_weights_path

def predict_mask_and_evaluate_model_on_test_data(model_weights_path):

    test_images_path_list = [os.path.join(test_path, image) for image in os.listdir(test_path)]
    print("test_images_path_list:", test_images_path_list)

    test_masks_path_list = [os.path.join(result_path, mask) for mask in os.listdir(test_path)]
    print("test_masks_path_list:", test_masks_path_list)

    model = load_model(model_weights_path)

    #model.compile(loss='binary_crossentropy', optimizer=SGD(lr=1e-4, momentum=0.9), metrics=['acc'])
    #model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['acc'])
    model.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.0001), metrics=['acc'])

    test_array = []
    mask_array = []
    for i in range(len(test_images_path_list)):
        test_img_path =  test_images_path_list[i]
        test_mask_path = test_masks_path_list[i]
        img = cv2.imread(test_img_path)
        img = img.astype("float32") / 255.0
        test_img = cv2.resize(cv2.imread(test_img_path), (224,224))
        test_img = test_img.astype("float32")
        test_img = test_img.reshape(1,224, 224,3)
        test_array.append(test_img)

        masked = model.predict(test_img)[0]
        mask_class0 = masked[:, :, 0]
        plt.imsave(test_mask_path, mask_class0)
        mask = cv2.resize(mask_class0, (224,224))
        mask= mask.reshape(1,224, 224, 1)
        mask_array.append(mask)
        plt.imshow(mask_class0)
        plt.show()

    print("metrics_name:", model.metrics_names)
    for i in range(len(test_array)):
        evaluation_test = model.evaluate(test_array[i], mask_array[i])
        test_loss = evaluation_test[0]
        test_accu = evaluation_test[1]
        print("loss of results:", test_loss)
        print("accuracy of results:", test_accu)



def main(): #Uncomment the two first line to train the model and the last line to predict the mask
    model = build_model()
    train_and_evaluate_model(A_List_of_Data_Path(), model)

if __name__ == "__main__":
    main()