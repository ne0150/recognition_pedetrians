//Face detection using HAAR Cascade
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using Emgu.CV;
using Emgu.CV.Structure;
using System;

public class peopleRecognition : MonoBehaviour
{
    public Slider people;
    public Slider people_n;

    public Slider body;
    public Slider body_n;

    public Slider upper;
    public Slider upper_n;

    public Dropdown drop;

    public Text people_t;
    public Text people_n_t;

    public Text bodys_t;
    public Text body_n_t;

    public Text upper_t;
    public Text upper_n_t;

    public Button bt;

    private int frameWidth;
    private int frameHeight;
    private VideoCapture cvCapture;

    private CascadeClassifier _cascadeClassifierPeople;
    private CascadeClassifier _cascadeClassifierFullBody;
    private CascadeClassifier _cascadeClassifierUpperBody;

    private Image<Bgr, byte> currentFrameBgr;

    public Material mt;

    void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        System.Collections.Generic.List<string> list = new System.Collections.Generic.List<string>();
        for (int i = 0; i < devices.Length; i++)
        {
            list.Add(i.ToString());
        }
        drop.AddOptions(list);

        bt.onClick.AddListener(StartFilming);

    }

    private void StartFilming()
    {
        started = true;

        cvCapture = new VideoCapture(drop.value);

        _cascadeClassifierPeople = new CascadeClassifier(Application.dataPath + "/pedestrians.xml");
        _cascadeClassifierFullBody = new CascadeClassifier(Application.dataPath + "/haarcascade_fullbody.xml");
        _cascadeClassifierUpperBody = new CascadeClassifier(Application.dataPath + "/haarcascade_upperbody.xml");

        frameWidth = (int)cvCapture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameWidth);
        frameHeight = (int)cvCapture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameHeight);

        try
        {
            cvCapture.Start();
        }
        catch(Exception e)
        {
            people_t.text = e.ToString();
        }
        
    }

    private bool started = false;
    void Update()
    {
        if (!started)
        {
            return;
        }

        faceDetector();
        UICotrol();
    }

    private void UICotrol()
    {
        people_t.text = people.value.ToString();
        people_n_t.text = people_n.value.ToString();

        bodys_t.text = body.value.ToString();
        body_n_t.text = body_n.value.ToString();

        upper_t.text = upper.value.ToString();
        upper_n_t.text = upper_n.value.ToString();
    }

    private void faceDetector()
    {
        
        currentFrameBgr = cvCapture.QueryFrame().ToImage<Bgr, byte>();

        Texture2D tex = new Texture2D(640, 480);

        if (currentFrameBgr != null)
        {
            Image<Gray, byte> grayFrame = currentFrameBgr.Convert<Gray, byte>();

            double scale_p = double.Parse(people_t.text);
            if(scale_p < 1.1 || scale_p > 2.0)
            {
                scale_p = 1.1;
            }

            int nei_p = int.Parse(people_n_t.text);
            if (nei_p < 1 || nei_p > 5)
            {
                nei_p = 1;
            }

            double scale_up = double.Parse(upper_t.text);
            if (scale_up < 1.1 || scale_up > 2.0)
            {
                scale_up = 1.1;
            }

            int nei_up = int.Parse(upper_n_t.text);
            if (nei_up < 1 || nei_up > 5)
            {
                nei_up = 1;
            }

            double scale_full = double.Parse(bodys_t.text);
            if (scale_full < 1.1 || scale_full > 2.0)
            {
                scale_full = 1.1;
            }

            int nei_full = int.Parse(body_n_t.text);
            if (nei_full < 1 || nei_full > 5)
            {
                nei_full = 1;
            }

            System.Drawing.Rectangle[] peoples = _cascadeClassifierPeople.DetectMultiScale(grayFrame, scale_p, nei_p, new System.Drawing.Size(frameWidth / 8, frameHeight / 8));
            System.Drawing.Rectangle[] upperbodys = _cascadeClassifierUpperBody.DetectMultiScale(grayFrame, scale_up, nei_up, new System.Drawing.Size(frameWidth / 8, frameHeight / 8));
            System.Drawing.Rectangle[] fullbodys = _cascadeClassifierFullBody.DetectMultiScale(grayFrame, scale_full, nei_full, new System.Drawing.Size(frameWidth / 8, frameHeight / 8));

            foreach (var man in peoples)
            {
                currentFrameBgr.Draw(man, new Bgr(0, 255, 0), 3);
            }

            foreach (var body_up in upperbodys)
            {
                currentFrameBgr.Draw(body_up, new Bgr(255, 0, 0), 3);
            }

            foreach (var body_full in fullbodys)
            {
                currentFrameBgr.Draw(body_full, new Bgr(0, 0, 255), 3);
            }

            //Convert this image into Bitmap, the pixel values are copied over to the Bitmap
            currentFrameBgr.ToBitmap();

            MemoryStream memstream = new MemoryStream();

            currentFrameBgr.Bitmap.Save(memstream, currentFrameBgr.Bitmap.RawFormat);

            tex.LoadImage(memstream.ToArray());
            mt.mainTexture = tex;
        }
    }

    private void OnDestroy()
    {
        //release from memory
        cvCapture.Dispose();
        cvCapture.Stop();

    }
}