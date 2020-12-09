import numpy as np
import glob
from PIL import Image
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Reference material:
# https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

# calculate frechet inception distance
def calculate_fid(model, real_images, fake_images):
	# calculate activations
	real_activations = model.predict(real_images)
	fake_activations = model.predict(fake_images)
	# calculate mean and covariance
	real_mu, real_sigma = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
	fake_mu, fake_sigma = fake_activations.mean(axis=0), np.cov(fake_activations, rowvar=False)
	# calculate sum squared difference between means
	sum_squared_difference = np.sum((real_mu - fake_mu) ** 2.0)
	# calculate sqrt of product between covariances
	covmean = sqrtm(real_sigma.dot(fake_sigma))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = sum_squared_difference + np.trace(real_sigma + fake_sigma - 2.0 * covmean)
	return fid

def load_images(path, image_size):
    image_paths = []
    for image_path in glob.glob(path + "*.png"):
        image_paths.append(image_path)
    image_paths.sort()

    images = np.empty((len(image_paths), image_size, image_size, 3))
    for i in range(len(image_paths)):
        image = Image.open(image_paths[i])
        images[i] = np.asarray(image) / np.float32(255.0)

    return images

def main():
    image_size = 256
    
    # initialize inception model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(256, 256, 3))

    for i in range(52):
        # load ground-truth images
        real_images = load_images("../../fid-scores/real/real-0-".format(i), image_size)
        # load generated images
        fake_images = load_images("../../fid-scores/fake/fake-{}-".format(i), image_size)

        # calculate fid score
        fid = calculate_fid(model, real_images, fake_images)
        print("Epoch {}, FID score: {}".format(i, fid))


if __name__ == '__main__':
    main()