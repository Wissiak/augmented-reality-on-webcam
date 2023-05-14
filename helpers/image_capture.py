import cv2

cap = cv2.VideoCapture(0)

idx = 0

name = input('Image name: ')

while(True):
    ret, frame = cap.read() 
    cv2.imshow('Input image', frame)
    
    pressedKey = cv2.waitKey(1) & 0xFF

    if pressedKey == ord('y'): #save on pressing 'y' 
        cv2.imwrite(f'images/{name}-{idx}.png',frame)
        cv2.destroyAllWindows()
        idx += 1
    elif pressedKey == ord('q'):
        break

cap.release()