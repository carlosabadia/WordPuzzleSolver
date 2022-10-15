# Functions used to find the word in order of use
xy_positionsvec2 = []


def find_word (wordsearch, word):
	"""Trys to find word in wordsearch and prints result"""
	# Store first character positions in array
	start_pos = []
	first_char = word[0]
	for i in range(0, len(wordsearch)):
		for j in range(0, len(wordsearch[i])):
			if (checktilde(wordsearch[i][j],first_char) or wordsearch[i][j] == '?'):
				start_pos.append([i,j])
	# Check all starting positions for word
	for p in start_pos:
		found,xy_positionsvec = check_start(wordsearch, word, p)
		if found:
			# Word foundf
			return xy_positionsvec,True
	# Word not found
	#print(word, ' No encontrada')
	return xy_positionsvec,False

def check_start (wordsearch, word, start_pos):
	"""Checks if the word starts at the startPos. Returns True if word found"""
	directions = [[-1,1], [0,1], [1,1], [-1,0], [1,0], [-1,-1], [0,-1], [1,-1]]
	# Iterate through all directions and check each for the word
	for d in directions:
		found,xy_positionsvec = check_dir(wordsearch, word, start_pos, d)
		if (found):
			return True,xy_positionsvec
	return False,xy_positionsvec

def check_dir (wordsearch, word, start_pos, dir):
	"""Checks if the word is in a direction dir from the start_pos position in the wordsearch. Returns True and prints result if word found"""
	xy_positionsvec = []
	found_chars = [word[0]] # Characters found in direction. Already found the first character
	current_pos = start_pos # Position we are looking at
	pos = [start_pos] # Positions we have looked at
	while (chars_match(found_chars, word)):
		if (len(found_chars) == len(word)):
			# If found all characters and all characters found are correct, then word has been found
			#print('')
			#print(word, ' Encontrada en:')
			#print('')
			# Draw wordsearch on command line. Display found characters and '-' everywhere else
			index =1 
			for x in range(0, len(wordsearch)):
				line = ""
				for y in range(0, len(wordsearch[x])):
					is_pos = False
					for z in pos:
						if (z[0] == x) and (z[1] == y):
							is_pos = True
					if (is_pos):
						xy_positions = {}
						xy_positions['x'] = x
						xy_positions['y'] = y
						xy_positionsvec.append(xy_positions)
						if (wordsearch[x][y] == '?'):
							if (dir[1]*dir[0] == -1):
								line = line + " " + word[len(word)-index]
							else:
								line = line + " " + word[index-1]
						else:
							line = line + " " + wordsearch[x][y]
						index = index + 1
					else:
						line = line + " -"
		
				#print(line)
			#print('')
			return True, xy_positionsvec
		# Have not found enough letters so look at the next one
		current_pos = [current_pos[0] + dir[0], current_pos[1] + dir[1]]
		pos.append(current_pos)
		if (is_valid_index(wordsearch, current_pos[0], current_pos[1])):
			found_chars.append(wordsearch[current_pos[0]][current_pos[1]])
		else:
			# Reached edge of wordsearch and not found word

			return False, xy_positionsvec 

	return False, xy_positionsvec 			


def chars_match (found, word):
	"""Checks if the leters found are the start of the word we are looking for"""
	index = 0
	for i in found:
		if ( (not checktilde(i,word[index])) and i != '?'):
			return False
		index += 1
	return True

def is_valid_index (wordsearch, line_num, col_num):
	"""Checks if the provided line number and column number are valid"""
	if ((line_num >= 0) and (line_num < len(wordsearch))):
		if ((col_num >= 0) and (col_num < len(wordsearch[line_num]))):
			return True
	return False

def checktilde (word1, word2):
	if (word1 == word2):
		return True
	if ((word1 == 'A' or word1 == 'Á') and (word2 == 'A' or word2 == 'Á')):
		return True
	if ((word1 == 'E' or word1 == 'É') and (word2 == 'E' or word2 == 'É')):
		return True
	if ((word1 == 'I' or word1 == 'Í') and (word2 == 'I' or word2 == 'Í')):
		return True
	if ((word1 == 'O' or word1 == 'Ó') and (word2 == 'O' or word2 == 'Ó')):
		return True
	if ((word1 == 'U' or word1 == 'Ú') and (word2 == 'U' or word2 == 'Ú')):
		return True
	if ((word1 == 'Ñ' or word1 == 'N') and (word2 == 'N' or word2 == 'Ñ')):
		return True
	return False