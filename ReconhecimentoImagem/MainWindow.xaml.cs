using Emgu.CV;
using Emgu.CV.Structure;
using System.Drawing;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace WpfApp2
{
	public partial class MainWindow : Window
	{
		private VideoCapture capture;
		private CascadeClassifier faceCascade;
		private bool isRunning = false;
		private DispatcherTimer timer;

		public MainWindow()
		{
			InitializeComponent();

			try
			{
				// Inicializa a captura de vídeo (câmera padrão)
				capture = new VideoCapture();

				// Carrega o classificador Haar Cascade para detecção facial
				faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml"); // Certifique-se de ter o arquivo XML

				// Inicializa o timer para atualizar a imagem
				timer = new DispatcherTimer();
				timer.Tick += ProcessFrame;
				timer.Interval = TimeSpan.FromMilliseconds(30); // Ajuste conforme necessário
				timer.Start();

				isRunning = true;
			}
			catch (Exception ex)
			{
				MessageBox.Show($"Erro ao inicializar: {ex.Message}");
			}
		}

		private void ProcessFrame(object sender, EventArgs e)
		{
			if (!isRunning) return;

			try
			{
				Mat frame = capture.QueryFrame();
				if (frame == null) return;

				Image<Bgr, byte> image = frame.ToImage<Bgr, byte>();

				// Detecta rostos na imagem
				Rectangle[] faces = faceCascade.DetectMultiScale(
					image,
					1.1,
					10,
					System.Drawing.Size.Empty);

				// Desenha retângulos verdes ao redor dos rostos detectados
				foreach (Rectangle face in faces)
				{
					image.Draw(face, new Bgr(Color.Green), 2);
				}

				// Converte a imagem para um formato que WPF pode exibir
				Bitmap bitmap = new Bitmap(frame.Width, frame.Height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
				System.Drawing.Imaging.BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, frame.Width, frame.Height), 
					 System.Drawing.Imaging.ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);

				int imageSize = frame.Width * frame.Height * 3; // 3 bytes por pixel (B, G, R)
				byte[] data = new byte[imageSize];
				System.Runtime.InteropServices.Marshal.Copy(image.MIplImage.ImageData, data, 0, imageSize);
				System.Runtime.InteropServices.Marshal.Copy(data, 0, bitmapData.Scan0, imageSize);

				bitmap.UnlockBits(bitmapData);

				var bitmapSource = Imaging.CreateBitmapSourceFromHBitmap(
					bitmap.GetHbitmap(),
					IntPtr.Zero,
					Int32Rect.Empty,
					BitmapSizeOptions.FromEmptyOptions());

				// Exibe a imagem no controle Image da WPF
				cameraImage.Source = bitmapSource;
			}
			catch (Exception ex)
			{
				MessageBox.Show($"Erro ao processar frame: {ex.Message}");
				StopCapture(); // Para a captura em caso de erro
			}
		}

		private void StopCapture()
		{
			if (timer != null)
			{
				timer.Stop();
				timer = null;
			}
			if (capture != null)
			{
				capture.Stop();
				capture.Dispose();
				capture = null;
			}
			if (faceCascade != null)
			{
				faceCascade.Dispose();
				faceCascade = null;
			}
			isRunning = false;
		}

		private void MainWindow_OnClosing(object sender, System.ComponentModel.CancelEventArgs e)
		{
			StopCapture();
		}
	}
}