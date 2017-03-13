import numpy as np
#np.seterr(divide='ignore', invalid='ignore')
from sklearn.cluster import KMeans


def make_cluster_mask_default(input_matrix,mask_image):
	
	[rows,cols]=mask_image.shape
	
	im_mask_flt = mask_image.flatten()
	find_loc = np.where(im_mask_flt==1)
	find_loc = list(find_loc)
	
	input_vector = input_matrix.flatten()
	
	input_select = input_vector[list(find_loc)]
	
	
	X = input_select.reshape(-1, 1)
	k_means = KMeans(init='k-means++', n_clusters=2)
	k_means.fit(X)
	k_means_labels = k_means.labels_
	k_means_cluster_centers = k_means.cluster_centers_
	k_means_labels_unique = np.unique(k_means_labels)
	

	
	# This is checked only for c15 ratio channel.
	center1 = k_means_cluster_centers[0]
	center2 = k_means_cluster_centers[1]
	
	
	if center1 < center2:
		# Interchange the levels.
		k_means_labels[k_means_labels == 0] = 99
		k_means_labels[k_means_labels == 1] = 0
		k_means_labels[k_means_labels == 99] = 1
		
	
	
	
	# 0 is sky and 1 is cloud
	cloud_pixels = np.count_nonzero(k_means_labels == 1)
	sky_pixels = np.count_nonzero(k_means_labels == 0)
	total_pixels = cloud_pixels + sky_pixels
	
	cloud_coverage = float(cloud_pixels)/float(total_pixels)
	
	# Final threshold image for transfer
	index = 0
	Th_image = np.zeros([rows,cols])
	for i in range(0,rows):
		for j in range(0,cols):
			
			if mask_image[i,j]==1:
				Th_image[i,j] = k_means_labels[index]
				index = index + 1			
	

	return(Th_image,cloud_coverage)	
