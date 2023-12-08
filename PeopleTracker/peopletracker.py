from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class PeopleTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        """
        Initialize a PeopleTracker to keep track of the next unique
        object ID, identified objects with their bounding rectangles
        and the number of consecutive frames that the object is
        marked as "disappeared".
        """
        self.nextObjectID = 0
        # {objectID: centroid}, centroid = (cX, cY)
        self.objects = OrderedDict()
        # {objectID: bounding_rectangle},
        # bounding_rectangle = [startX, startY, endX, endY]
        self.boxes = OrderedDict()
        # {objectID: num of consecutive frames
        # that the object is marked as disappeared}
        self.disappeared = OrderedDict()

        # the number of maximum consecutive frames a given object
        #  is allowed to be disappeared until we need to unregister
        #  the object from tracking
        self.maxDisappeared = maxDisappeared

        # the maximum distance between centroids to associate an object
        #  if the distance is larger than max distance, mark the object
        #  as "disappeared"
        self.maxDistance = maxDistance

    def register(self, centroid, box):
        """
        Register an object to store it's centroid and bounding rectangle.
        """
        self.objects[self.nextObjectID] = centroid
        self.boxes[self.nextObjectID] = box
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def unregister(self, objectID):
        """
        Unregister an object from the tracker.
        """
        del self.objects[objectID]
        del self.boxes[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # if no object is detected or being tracked
        if len(rects) == 0:
            # mark any existing tracked objects as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # unregister any disappeared objects that have reached
                # the max number of consecutive frames
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.unregister(objectID)

            # return nothing since nothing is detected or tracked
            return self.objects, self.boxes

        # else (there is object in the current frame)
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for i, (startX, startY, endX, endY) in enumerate(rects):
            # calculate centroid from a bounding rectangle
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects, register every
        # objects in the list
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
                
		# else, we need to check if the input object already existed
		# in the tracker
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

			# compute the Euclidean distance between each pair of 
			# existing centroids and input centroids
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
			# sort the row indexes 
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or unregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for row, col in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # if the distance between centroids is greater than
                # the maximum distance, do not associate the two
                # centroids to the same object
                if D[row, col] > self.maxDistance:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.boxes[objectID] = rects[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants unregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.unregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col])

        # return the set of trackable objects
        return self.objects, self.boxes
