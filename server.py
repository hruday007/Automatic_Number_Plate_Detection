from flask import *
from main import *
import cv2

import datetime
import pandas as pd
from csv import writer


app = Flask(__name__)
plates = set()

def gen_frames():
    findPlate = PlateFinder()

    # Initialize the Neural Network
    model = NeuralNetwork()

    cap = cv2.VideoCapture('video.MOV')
    while (cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            # cv2.imshow('original video', img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            # cv2.waitKey(0)
            possible_plates = findPlate.find_possible_plates(img)
            if possible_plates is not None:
                for i, p in enumerate(possible_plates):
                    chars_on_plate = findPlate.char_on_plate[i]
                    recognized_plate, _ = model.label_image_list(chars_on_plate, imageSizeOuput=128)
                    # print(recognized_plate)
                    # plates.add(recognized_plate)
                    x = datetime.datetime.now()
                    y = x.strftime("%c")
                    # headerList = ['Number_plate', 'Timestamp']
                    list =[]
                    list.append(recognized_plate)
                    list.append(y)
                    with open('plates.csv', 'w') as fp:
                        # fp.write(recognized_plate)
                        # fp.write(y)
                        wrt_obj = writer(fp)
                        # wrt_obj.writerow(headerList)
                        wrt_obj.writerow(list)
                        

                        
                    # cv2.imshow('plate', p)

                    cv2.imwrite('./static/img/plate.jpg', p)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break


        else:
            break
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_plates():
    if len(plates) > 0:
        yield ''
    else:
        yield list(plates)[-1]

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plate_feed')
def plate_feed():
    return Response(gen_plates(), mimetype='text/plain')

@app.route('/file/<filename>')
def send_file(filename):
    print(filename)
    return send_from_directory('./', filename)

@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/')
def hello_world():
    return render_template('index.html')




if __name__ == '__main__':
    with open('plates.csv', 'w') as fp:
        pass
    app.run(debug=True) 