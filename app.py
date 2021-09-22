from flask import  Flask, render_template,Response
from camera import Video_Emotion_1, Video_Sign, Video_Emotion_2
global video1, video2, video3


app = Flask(__name__)
def gen(camera):
    while True:
      frame=camera.get_frame()
      print("camera reading....")
      yield(b'--frame\r\n'
    b'Content-Type: image/jpeg\r\n\r\n'+frame+
    b'\r\n\r\n')
      print("cameraReading")


#----------------home page--------------
@app.route("/")
def index():
    return render_template('index.html')

#--------------------emotion channel 1-----------
@app.route("/emotion_video_2")
def output_emotion_1():
    return render_template('output_emotion_1.html')

#--------------------emotion video 1-----------
@app.route("/video_emotion_2")
def video_emotion_1():
    return Response(gen(Video_Emotion_1()), mimetype='multipart/x-mixed-replace; boundary=frame')

#--------------------emotion channel 2-----------
@app.route("/emotion_video_1")
def output_emotion_2():
    return render_template('output_emotion_2.html')

#--------------------emotion video 2-----------
@app.route("/video_emotion_1")
def video_emotion_2():
    return Response(gen(Video_Emotion_2()), mimetype='multipart/x-mixed-replace; boundary=frame')

#--------------------Sign  channel-----------
@app.route("/sign_video")
def output_sign():
    return render_template('output_sign.html')

#--------------------Sign video-----------
@app.route("/video_sign")
def video_sign():
    return Response(gen(Video_Sign()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True,port=1000)


