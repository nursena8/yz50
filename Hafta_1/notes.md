# Hafta 1
# Bu Hafta Öğrendiklerim

**Kaynaklar:**
- 3Blue1Brown  But what is a neural network? (https://www.youtube.com/watch?v=aircAruvnKk)
- 3Blue1Brown  Gradient descent, how neural networks learn (https://www.youtube.com/watch?v=IHZwWFHWa-w)
- Andrej Karpathy  The spelled-out intro to neural networks and backpropagation ilk 19 dakika (https://www.youtube.com/watch?v=VMj-3S1tku0)

---

## 1. Neural Network 

- Bir sinir ağı katmanlara dizilmiş nöronlardan oluşur input layer, hidden layer, output layer. 
- Bir nöronun ham aktivasyondan önceki çıktısı `z = w*x+b` sınırsız bir sayıdır belirli bir aralığa sıkışmaz.
- Bu `z` değeri bir aktivasyon fonksiyonundan geçirilerek belirli bir aralığa sıkıştırılır.
- Sigmoid aktivasyonundan geçen bir değer 0 ile 1 arasına sıkışır bu sadece sigmoide özel bir davranıştır.
- Her nöron kendi ağırlıklarına ve kendi bias ine  sahiptir.
- Bir nöronun aldığı girdi sayısı o nöronun sahip olduğu ağırlık sayısına eşittir.
- Bir sonraki katmandaki her nöron bir önceki katmandaki tüm  nöronlardan girdi alır.
- Ham değer formülü tek girdili bir nöron için: `z = w * x + b`
- Çok girdili bir nöron için: `z = w1*x1 + w2*x2 + ... + wn*xn + b`

### Örnek  — 1 girdili, 1 ağırlıklı, tek nöron
- `z = w * x + b`
- Burada nöronun sadece 1 girdisi (`x`), 1 ağırlığı (`w`) ve 1 bias'ı (`b`) var

### Örnek  — 2 girdili, 2 ağırlıklı, tek nöron
- `z = w1*x1 + w2*x2 + b`

### Örnek — 2 girdili, 2 nöronlu katman
- İki nöron da aynı girdileri (`x1, x2`) görür, ama her nöronun kendi ağırlıkları ve kendi bias'ı vardır
- Nöron 1: `z1 = w1_1*x1 + w1_2*x2 + b1`
- Nöron 2: `z2 = w2_1*x1 + w2_2*x2 + b2`
- Katmanın çıktısı 2 sayılık bir liste olur: [out1, out2]

### Örnek — 2 girdili, 3 nöronlu katman
- 3 nöron da aynı [x1, x2] girdisini görür
- Her nöronun kendine ait 2 ağırlığı  ve 1 bias'ı vardır
- Nöron 1: `z1 = w1_1*x1 + w1_2*x2 + b1`
- Nöron 2: `z2 = w2_1*x1 + w2_2*x2 + b2`
- Nöron 3: `z3 = w3_1*x1 + w3_2*x2 + b3`
- Katmanın çıktısı 3 sayılık bir liste olur: `[out1, out2, out3]`

---
Ilk başta anlamadığım için bu örnekler çok işe yarıyor :)))

## 2. Aktivasyon Fonksiyonları

- Nöronun ham çıktısı (`z`) sınırsız bir sayı çok büyük pozitif veya çok büyük negatif olabilir.
- Bu değeri belirli bir aralığa sıkıştırmak için aktivasyon fonksiyonundan geçiriyoruz.
- Aktivasyon fonksiyonu olmadan art arda dizilen katmanlar sadece doğrusal işlemler üst üste binmiş olurdu ve ağ karmaşık ilişkiler öğrenemezdi.
- **Sigmoid:** 1 / (1 + e^-z) çıktıyı 0 ile 1 arasına sıkıştırır.
- **Tanh:** çıktıyı -1 ile 1 arasına sıkıştırır sigmoid'e benzer ama merkezi sıfırdadır. 
- **ReLU (Rectified Linear Unit):** max(0, z), yani z negatifse 0, pozitifse z nin kendisi
- ReLU gerçek hayatta  daha çok tercih edilir, çünkü:
  - Sigmoid, z çok büyük ya da küçük olduğunda neredeyse düzleşiyor bu da gradyanın çok küçülüp öğrenmeyi yavaşlatmasına sebep oluyor.
---

## 3. Loss Fonksiyonları

- Loss neural networkün tahmininin hedeften ne kadar uzak olduğunu tek bir sayıya indirger.
- Amaç: eğitim boyunca bu sayıyı mümkün olduğunca sıfıra yaklaştırmak.
- **Tahmin edilen değer (`y_pred`)** aktivasyon fonksiyonundan geçirilmiş değerdir yani sigmoid(z) veya relu(z) sonrası elde edilen sayı.
- **MSE (Mean Squared Error):** (y_true - y_pred)^2, birden fazla örnek varsa bunların ortalaması alınır.
- MSE de fark neden kareleniyor:
  - İşaretin + , - önemini ortadan kaldırmak için.
  - Büyük hataları küçük hatalardan daha çok cezalandırmak için.
- **MAE (Mean Absolute Error):** farkın mutlak değerinin ortalaması büyük hataları MSE kadar sert cezalandırmaz.
- **Cross-Entropy Loss:** özellikle sınıflandırma problemlerinde kullanılır tahmin edilen olasılık dağılımı ile gerçek etiket arasındaki farkı ölçer.

---

## 4. Manuel Ağırlık Denemesi Gradient Descenti kavramamız için önemli bir örnek.

- Gradient descente geçmeden önce `w` değerini elle değiştirip lossun nasıl değiştiğini gözlemledim.
- Bu deneme lossun neye bağlı olduğunu ağırlığa, girdiye, biasa, hedefe olan uzaklığı kavramamı sağladı.
- parametre sayısı arttıkça gerçek ağlarda milyonlarca parametre olduğu için elle tarama imkansız hale geliyor bu yüzden gradient descent yönetmine ihtiyacımız var. 

---

## 5. Sayısal Türev (Numerical Derivative)

- Karpathy'nin yaptığı yöntem (forward difference): sadece `h` kadar sağa kaydırıp loss'un ne kadar değiştiğine bakıyor
  - `f'(w) = (f(w+h) - f(w)) / h`
- Araştırdığım merkezi fark (central difference) yöntemi: hem sağa hem sola bakıp ikisi arasındaki farkı `2h`'ye bölüyor
  - `f'(w) = (f(w+h) - f(w-h)) / (2h)`
- İki yöntem arasında pratikte gözle görülür bir fark olmadığını gözlemledim
- Bunun sebebi `h` ı  çok küçük seçmemiz (`h=0.0001` gibi)
- `h` küçük seçilince baktığımız aralık da küçülüyor bu sayede ölçtüğümüz eğim gerçek türeve daha çok yaklaşıyor
- `h` ı  aşırı küçültmemek gerekiyor çünkü floating point  hassasiyet hatası oluşabiliyor
- Araştırma sorusu: hangi durumlarda `h` büyük seçilir ? 
  - Gerçek dünya ölçümlerinde veya simülasyon sonuçlarında gürültü varsa
  - Böyle durumlarda `h` ı  büyük seçmek, gürültüyü asıl sinyalden ayırt etmeye yardımcı olabiliyor.

---

## 6. Gradient Descent

- Amaç: ağırlıkları optimize ederek loss'u minimize etmek
- Gradyan bize iki şeyi birden söylüyor:
  - Yön:`w`yi artırmak mı azaltmak mı gerektiğini
  - Şiddet: ne kadar sert/büyük bir adım atmamız gerektiğini
- Gradyan sayesinde `w`yi rastgele denemek yerine bilinçli hesaplanmış bir şekilde optimize edebiliyoruz.
- Güncelleme formülü: `w = w - learning_rate * grad`
  - Bu ifadenin amacı: `w` yi loss'u azaltacak yönde küçük bir adım kadar güncellemek.
  - Gradyan matematiksel olarak "loss'un arttığı" yönü gösteriyor, bu yüzden çıkarma işlemi yapıyoruz amacımız lossu azaltmak.
  - `grad` pozitifse → `w` yi azalt
  - `grad` negatifse → `w` yi artır

- 3Blue1Brown'ın "gradient descent" videosundan:
  - Loss  fonksiyonu tüm ağırlık ve biaslara bağlı çok boyutlu bir yüzey olarak düşünülebilir.
  - Backpropagation bu gradyanı büyük ağlarda verimli şekilde hesaplamanın yöntemidir sayısal türev yerine kullanılır.
  - Stokastik gradient descent: tüm veri seti yerine küçük gruplar üzerinden gradyan hesaplayarak eğitimi hızlandırma yöntemi.

---

## 7. Karpathy Videosu — İlk 19 Dakika 

- Micrograd, bir autograd otomatik gradyan motoru ve backpropagation'ı uyguluyor.
- Backpropagation loss'un ağırlıklara göre gradyanını verimli şekilde hesaplayan algoritma.
- PyTorch  gibi kütüphanelerin matematiksel çekirdeği de aynı mantığa dayanıyor
- Micrograd skaler tek tek sayı düzeyinde çalışıyor tensörler kadar verimli değil ama backpropu anlamak için önemli zaten karpathy de potansiyel olarak ders içeriği olarak kullanılabildiğini söylüyor.
- Türevin klasik tanımı: lim(h→0) [f(x+h) - f(x)] / h
- Sembolik türev yerine sayısal türev  kullanılıyor çünkü gerçek ağlarda ifadeler elle türetilemeyecek kadar karmaşık.
- Sayısal türev yavaş ve her parametre için fonksiyonu tekrar tekrar hesaplamak gerekiyor.
- Sayısal türev h seçimine bağlı hassasiyet sorunları taşıyor.
- Analitik türev, h gibi bir yaklaşıklığa gerek duymadan tam ve kesin sonucu doğrudan hesaplıyor.