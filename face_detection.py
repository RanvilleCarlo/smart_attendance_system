# importing librarys
import cv2
import numpy as npy
import face_recognition as face_rec
# function
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)


# img declaration
ranville = face_rec.load_image_file('D:\\smart_attendance_system\\sample_images\\ranville.jpg')
ranville = cv2.cvtColor(ranville, cv2.COLOR_BGR2RGB)
ranville = resize(ranville, 0.50)
ranville_test = face_rec.load_image_file('D:\\smart_attendance_system\\sample_images\\ranville_test.jpg')
ranville_test = resize(ranville_test, 0.50)
ranville_test = cv2.cvtColor(ranville_test, cv2.COLOR_BGR2RGB)

# finding face location

faceLocation_ranville = face_rec.face_locations(ranville)[0]
encode_ranville = face_rec.face_encodings(ranville)[0]
cv2.rectangle(ranville, (faceLocation_ranville[3], faceLocation_ranville[0]), (faceLocation_ranville[1], faceLocation_ranville[2]), (255, 0, 255), 3)


faceLocation_ranvilletest = face_rec.face_locations(ranville_test)[0]
encode_ranvilletest = face_rec.face_encodings(ranville_test)[0]
cv2.rectangle(ranville_test, (faceLocation_ranville[3], faceLocation_ranville[0]), (faceLocation_ranville[1], faceLocation_ranville[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_ranville], encode_ranvilletest)
print(results)
cv2.putText(ranville_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )

cv2.imshow('main_img', ranville)
cv2.imshow('test_img', ranville_test)
cv2.waitKey(0)
cv2.destroyAllWindows()