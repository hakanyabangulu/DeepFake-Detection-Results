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
		
	


****
