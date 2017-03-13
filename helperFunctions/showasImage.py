import numpy as np

def showasImage(input_matrix):
	
	is_matrix = len(input_matrix.shape)
	
	if is_matrix == 2: # Input is a 2D numpy array
		
		
		(rows,cols) = input_matrix.shape
		
		output_matrix = np.zeros([rows,cols])
		
		inp_flat = input_matrix.flatten()
		
		min_value = np.amin(inp_flat)
		max_value = np.amax(inp_flat)
		
		diff = max_value - min_value
		
		for i in range(0,rows):
			for j in range(0,cols):
				output_matrix[i,j] = ((input_matrix[i,j] - min_value)/diff)*255
				
	else:	# Input is a 1D numpy array
		(rows) = input_matrix.size
		output_matrix = np.zeros(rows)
		
		inp_flat = input_matrix
		
		min_value = np.amin(inp_flat)
		max_value = np.amax(inp_flat)
		
		diff = max_value - min_value
		
		for i in range(0,rows):
			output_matrix[i] = ((input_matrix[i] - min_value)/diff)*255

	
	return(output_matrix)
