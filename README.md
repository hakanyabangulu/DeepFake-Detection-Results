# DeepFake-Detection
For CNN : 

	* General Features : 
		* BATCH = 32
		* EPOCHS = 10
		* LR = 1e-3
		* FRAMES_PER_VIDEO = 5
		* THRESHOLD = 0.5
		* FRAME_SAMPLE_RATE = 5
		* CONFIDENCE_THRESHOLD = 0.5

! First Features
  
	CONFIDENCE_THRESHOLD = 0.5
	FRAMES_PER_VIDEO = 5
	FRAME_SAMPLE_RATE = 5 ayarlandı.
	Toplamda 205 Real & 205 Fake video ile Extraction işlemi yapıldı.
	! 1026 Real & 1026 Fake yüz bulabildi.
	
		Test Results -> Accuracy: 0.9300, Average Loss: 0.3900, Precision: 0.9161, Recall: 0.9467, F1-Score: 0.9311 = > Yunet
		
		Test Results -> Accuracy: 0.9300, Average Loss: 0.4452, Precision: 0.9329, Recall: 0.9267, F1-Score: 0.9298 = > MTCNN
		
		Test Results -> Accuracy: 0.6967, Average Loss: 0.6034, Precision: 0.8734, Recall: 0.4600, F1-Score: 0.6026 = > Haar Cascade

		! Son 5 video çıkarıldı. 145 Real 145 Fake :
		
		Test Results -> Accuracy: 0.9552, Average Loss: 0.2206, Precision: 0.9342, Recall: 0.9793, F1-Score: 0.9562 = > Yunet
		
		Test Results -> Accuracy: 0.9367, Average Loss: 0.4081, Precision: 0.9645, Recall: 0.9067, F1-Score: 0.9347 = > MTCNN
		
	= > Extraction işlemi için Yunet daha iyi sonuç verdiği için Yunet ile devam ettim.	
	
! Second Features

	CONFIDENCE_THRESHOLD = 0.95
	Sample Rate & Frames Per Video sabit bırakıldı.
	* Amaç net yüzlerle modeli eğitmek. Bunu yaptığım zaman bazı videolarda yüz bulmakta zorlandı. Bazılarında 5 bazılarında 1 bazılarında 0 buldu.
	Toplamda 205 Real & 205 Fake video ile Extraction işlemi yapıldı.
	! 687 Real & 687 Fake yüz bulabildi.
	! Test videoları için CONFIDENCE_THRESHOLD = 0.6 ya çekildi.
	! Eğitim sonuçları : (Validation Accuracy=0.9382, Validation Loss=0.1955)
	
		Test Results -> Accuracy: 0.8500, Average Loss: 0.7637, Precision: 0.8621, Recall: 0.8333, F1-Score: 0.8475
		
	! Test sonuçları ilkine kıyasla kötü olduğu için CONFIDENCE_THRESHOLD = 0.9 'a yükseltildi.
	
		Test Results -> Accuracy: 0.8500, Average Loss: 0.7637, Precision: 0.8621, Recall: 0.8333, F1-Score: 0.8475
		
	! Test sonuçları aynı bu yüzden son olarak CONFIDENCE_THRESHOLD = 0.94 'e yükseltildi. 0.95 'te yüz bulmakta zorlandı bu yüzden 0.94 e ayarlandı. 0.95 e kıyasla test daha hızlı,
	diğer değerlere göre daha yavaş gerçekleşti ama her videodan 5 frame almayı başardı.
		
		Test Results -> Accuracy: 0.8456, Average Loss: 0.6021, Precision: 0.8592, Recall: 0.8243, F1-Score: 0.8414
		
! Third Features

		* EPOCHS = 15
		* FRAMES_PER_VIDEO = 8
		* CONFIDENCE_THRESHOLD = 0.94 ayarlandı. Validation set sonunda f1 skora göre test için otomatik seçiliyor. 
		* Confidence hesaplanırken artık mean yerine median kullanılıyor. 
		
		! 1600 Real & 1600 Fake yüz bulabildi.
		! Eğitim sonuçları :  (ValAcc=0.9548, ValLoss=0.1050)
		
			Test Results -> Accuracy: 0.9463, Average Loss: 0.5863, Precision: 0.9459, Recall: 0.9459, F1: 0.9459
		! Kod üzerinde küçük değişiklikler yapıldı. Ekstra olarak Confusion Matrix eklendi.
		
			Test Results -> Accuracy: 0.8926, Average Loss: 0.5019, Precision: 0.9394, Recall: 0.8378, F1: 0.8857
			Ekstradan 360 adet başka ve daha zor fake videolar ile test edildi ve sonuçlar çok fazla düştü.
			Not: Yapılan değişiklikler daha kötü sonuç verdiği için Confusion Matrix harici işlemler geri alındı.

! Fourth Features

		* FRAMES_PER_VIDEO = 10
		* Extraction işleminde fake etiketli videolar için IOU kullanıldı.
		* Real etiketliler için kullanılmadı.
		
		! Bu sefer Extraction işlemi için farklı videolar (anlaşılması daha zor olan ve çoğunlukla 2 veya 3 kişi bulununan videolar) kullanıldı.
		! Fake videolar extraction işlemi sırasında takip edildi, çoğunlukla net ve fake olduğu belirli olan yüzlerin çekildiği fark edildi.
		! Ama bazı yüzlerin fake olsa bile real gibi göründüğü fark edildi.
		! Toplamda 205 Real & 205 Fake videoda Extraction işlemi yapıldı.
		! 2050 Real & 1196 Fake yüz bulabildi.
		! Beklenildiği üzere Fake videolardan alınan yüzler Real videolardan alınan yüzlerden az çıktı.
		! Dengeyi sağlamak adına eğitime başlamadan önce Real videolardan net olmayan yüzler çıkartıldı ve fake videolardan biraz daha eklendi.
		! 1800 Real & 1800 Fake yüz ile eğitim yapıldı.
		
		! Eğitim Sonuçları : (ValAcc=0.9722, ValLoss=0.0872)
			
			Test Results -> Accuracy: 0.6167, Average Loss: 2.0670, Precision: 0.7536, Recall: 0.3467, F1: 0.4749
			Not: Burda beklenen şey zor olan videolardaki tahminlerin doğru olması yönündeydi. Eğitim sonuçları iyi şeyler vaat ediyordu.
			Bazı zor olan videolardaki sonuçlar incelendi. Modelin eğitilmesi sürecinde herhangi bir sorunun olmadığı , sorunun veri setinde olduğu anlaşıldı.
			Sonuç: Zor olan videolar insanın bile anlayamayacağı videolardı. Bu yüzden test için videolar değiştirildi.
			
			
! Fifth Features

		* BATCH = 32
		* EPOCHS = 15
		* LR = 1e-4
		* FRAMES_PER_VIDEO = 10
		* THRESHOLD = 0.5
		* FRAME_SAMPLE_RATE = 5
		* CONFIDENCE_THRESHOLD = 0.94
		
		! Veri seti karıştırıldı. Real için Yunet , Fake için Mtcnn kullanıldı.
		! 150 Real & 175 Fake video test edildi.
			
			Test Results -> Accuracy: 0.9569, Average Loss: 0.2141, Precision: 0.9600, Recall: 0.9600, F1: 0.9600
			
<img width="333" height="266" alt="Threshold = 0 5" src="https://github.com/user-attachments/assets/edcdd3b7-7ef3-415e-a2f5-737062d195a0" />
			
			Not: 150 Real videodan 7 adet kaçırdı , 175 Fake videodan 7 adet kaçırdı.
			Sonuç: Bu sonuçlar modelin iyi eğitildiği ve veri setinin iyi olduğu anlamına geliyor. Son olarak veri seti genişletilecek ve bu özelliklerle son bir test işlemi yapılacak. 	
		
	


****
