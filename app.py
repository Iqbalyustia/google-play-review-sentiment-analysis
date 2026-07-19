import streamlit as st
import pandas as pd
from google_play_scraper import reviews, Sort
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Download necessary data for nltk
nltk.download('vader_lexicon')

# Kata kunci untuk masing-masing domain
keywords_similarity = {
    'Performance': [
        'mudah', 'sederhana', 'simpel', 'simple', 'lancar', 'easy', 'accessible', 'aksesibilitas', 'keterjangkauan', 'mdh',
        'ringan', 'gampang', 'mudah diakses', 'mudah digunakan', 'akses mudah', 'navigasi mudah', 'mudahan', 'mudh',
        'cepat', 'kilat', 'segera', 'fast', 'cepat mudah', 'cepat tanggap', 'cepat responsif', 'pantas', 'cpt', 'cepat merespon',
        'cepat diselesaikan', 'respon cepat', 'penyelesaian cepat', 'cepatan', 'cpat', 'baik', 'bagus', 'prima', 'optimal',
        'baik sesuai', 'sesuai dengan yang baik', 'kualitas baik', 'kinerja baik', 'good', 'bgus', 'bagus', 'kualitas tinggi',
        'layanan baik', 'baikan', 'mantap', 'sesuai', 'cocok', 'tepat', 'pas', 'sinkron', 'dipahami', 'dimengerti', 'dipelajari',
        'dipahami', 'understandable', 'mudah dipahami', 'sederhana dipahami', 'jelas dimengerti', 'easy to understand',
        'mdh dimengerti', 'easily understood', 'mudah dimengerti', 'jelas dipahami', 'terstruktur dengan baik', 'instruksi yang jelas',
        'mudah dipaham', 'efisien', 'efficient', 'efektif', 'hemat', 'produktif', 'efis', 'hemat waktu', 'efisien tinggi',
        'waktu penggunaan yang efisien', 'penggunaan sumber daya yang hemat', 'efisiennya', 'tepat waktu hasil', 'hasil yang tepat waktu',
        'Meningkatkan', 'enhance', 'peningkatan', 'pembaharuan', 'menaikkan', 'memperbaiki', 'meningkat', 'improve', 'meningkatkan produktivitas',
        'meningkatkan efisiensi', 'meningkatkan efektivitas', 'meningkatkan kualitas', 'mgkat', 'stabil', 'tetap', 'kokoh', 'pengguna',
        'penggunaan', 'tampilan', 'penampilan', 'hasil', 'output', 'hasil produksi', 'hasil kerja', 'output', 'hsl', 'hasil akhir', 'output kerja',
        'hasil yang diharapkan', 'hasil yang diinginkan', 'hasil-hasil', 'kepuasan', 'puas', 'satisfaction', 'satisfying', 'memenuhi harapan',
        'kepuasan', 'kepuasan pengguna', 'kepuasan pelanggan', 'memuaskan', 'memuaskan hati', 'memberikan kepuasan', 'mgkatkan ksnsn',
        'memadai', 'meningkatkan puas', 'Meningkatkan kepuasan', 'meningkatkan kepuasan pelanggan', 'kepuasan pengguna yang ditingkatkan',
        'meningkatkan kegembiraan', 'memperbaiki kepuasan', 'menaikkan kepuasan', 'improve satisfaction', 'meningkatkan kepuasan pelanggan',
        'meningkatkan kualitas layanan', 'memuaskan pelanggan', 'memberikan kepuasan', 'meningkatkan loyalitas', 'meningkatkan retensi pelanggan',
        'fitur', 'fiturnya', 'meningkatkan kepuasan-kepuasan', 'pengalaman pengguna yang memuaskan', 'layanan yang memuaskan', 'kepuasan pelanggan yang berkelanjutan',
        'kepuasan pengguna yang tinggi', 'memuaskan-memuaskan', 'memuaskn', 'mgkatkan ksnsn', 'msksn', 'memenuhi', 'puaskan', 'kegembiraan', 'kesenangan',
        'kepuasan hati'
    ],
    'Information': [
        'informasi', 'info', 'data', 'pengetahuan', 'berita', 'knowledge', 'informative', 'informing', 'informed', 'information accuracy',
        'data integrity', 'data accuracy', 'info update', 'up-to-date', 'reliable information', 'accurate information', 'info lengkap', 'complete information',
        'comprehensive information', 'informasi detail', 'detailed information', 'clarity of information', 'clear information', 'info jelas', 'jelas',
        'transparency', 'transparan', 'transparency of information', 'user-friendly information', 'informasi user-friendly', 'info terkini', 'current information',
        'timely information', 'info tepat waktu', 'tepat waktu', 'data relevan', 'relevant data', 'relevant information', 'info relevan', 'data terbaru',
        'latest data', 'up-to-date data', 'informasi terbaru', 'latest information', 'informasi penting', 'important information', 'data analysis', 'analisis data',
        'data-driven', 'data analysis tools', 'alat analisis data', 'data-driven decisions', 'keputusan berdasarkan data', 'info yang tepat', 'informasi yang tepat',
        'tepat guna', 'fit for purpose information', 'information for decision-making', 'informasi untuk pengambilan keputusan', 'data science', 'ilmu data', 'data mining',
        'penambangan data', 'informasi terpercaya', 'trusted information', 'secure information', 'info aman', 'data protection', 'perlindungan data', 'data privacy',
        'privasi data', 'confidential information', 'informasi rahasia', 'data security', 'keamanan data', 'data governance', 'tata kelola data', 'information governance',
        'governance', 'regulasi informasi', 'informasi yang valid', 'valid information', 'verifiable information', 'info yang bisa diverifikasi', 'verifikasi data',
        'data verification', 'informasi akurat', 'akurasi informasi', 'data quality', 'kualitas data', 'quality information', 'informasi berkualitas', 'data quality control',
        'kontrol kualitas data', 'info yang valid', 'valid data', 'data analysis', 'analisis informasi', 'data insights', 'wawasan data', 'informasi berguna', 'useful information',
        'informasi bermanfaat', 'beneficial information', 'informasi yang berguna', 'reliable information', 'informasi yang andal', 'data relevan', 'data up-to-date', 'updated information',
        'info terbaru', 'informasi terkini', 'up-to-date info', 'up-to-date information', 'informasi real-time', 'real-time information', 'informasi cepat', 'quick information',
        'data processing', 'pengolahan data', 'info processing', 'pengolahan informasi', 'data analytics', 'analitik data', 'informasi analitik', 'analytic information', 'info analitik',
        'analisis informasi'
    ],
    'Economy': [
        'ekonomi', 'hemat', 'biaya', 'murah', 'terjangkau', 'affordable', 'cost-effective', 'cost efficiency', 'efisiensi biaya', 'biaya rendah',
        'low cost', 'penghematan biaya', 'cost savings', 'penghematan', 'saving', 'budget', 'anggaran', 'cost control', 'kontrol biaya', 'spending',
        'pengeluaran', 'biaya efektif', 'effective cost', 'cost benefit', 'biaya manfaat', 'benefit-cost ratio', 'rasio biaya manfaat', 'nilai uang',
        'value for money', 'economic value', 'nilai ekonomi', 'return on investment', 'ROI', 'pengembalian investasi', 'investasi', 'investment',
        'economic impact', 'dampak ekonomi', 'financial', 'keuangan', 'finansial', 'financial management', 'manajemen keuangan', 'financial planning',
        'perencanaan keuangan', 'financial savings', 'penghematan keuangan', 'penghematan finansial', 'economic performance', 'kinerja ekonomi',
        'economic efficiency', 'efisiensi ekonomi', 'economic growth', 'pertumbuhan ekonomi', 'growth', 'pertumbuhan', 'economic benefits',
        'manfaat ekonomi', 'economic analysis', 'analisis ekonomi', 'economic factors', 'faktor ekonomi', 'factors', 'economic stability', 'stabilitas ekonomi',
        'stability', 'ekonomi stabil', 'economic trends', 'tren ekonomi', 'trends', 'market trends', 'tren pasar', 'economic conditions', 'kondisi ekonomi',
        'economic environment', 'lingkungan ekonomi', 'economic climate', 'iklim ekonomi', 'business climate', 'iklim bisnis', 'market', 'pasar',
        'market conditions', 'kondisi pasar', 'market analysis', 'analisis pasar', 'market performance', 'kinerja pasar', 'economic indicators', 'indikator ekonomi',
        'economic policies', 'kebijakan ekonomi', 'policies', 'policy', 'regulations', 'regulasi', 'economic regulation', 'regulasi ekonomi', 'fiscal policy',
        'kebijakan fiskal', 'monetary policy', 'kebijakan moneter', 'tax', 'pajak', 'taxation', 'perpajakan', 'tariffs', 'tarif', 'subsidies', 'subsidi',
        'economic development', 'pembangunan ekonomi', 'development', 'economic planning', 'perencanaan ekonomi', 'economic management', 'manajemen ekonomi',
        'cost efficiency', 'efisiensi biaya', 'business performance', 'kinerja bisnis', 'profit', 'profitabilitas', 'profitability', 'keuntungan', 'laba',
        'economic strategy', 'strategi ekonomi', 'strategy', 'strategi', 'competitive advantage', 'keunggulan kompetitif', 'competitive', 'kompetitif',
        'economic strength', 'kekuatan ekonomi', 'strength', 'economic opportunities', 'peluang ekonomi', 'opportunities', 'economic challenges', 'tantangan ekonomi',
        'challenges', 'opportunities and challenges', 'peluang dan tantangan', 'economic landscape', 'landskap ekonomi', 'landscape', 'economic resources', 'sumber daya ekonomi',
        'resources', 'economic resilience', 'ketahanan ekonomi', 'resilience', 'resilient economy', 'ekonomi yang tangguh', 'financial health', 'kesehatan keuangan',
        'economic efficiency', 'efficiency', 'economic sustainability', 'keberlanjutan ekonomi', 'sustainability', 'economic equity', 'keadilan ekonomi', 'equity', 'ekuitas',
        'economic mobility', 'mobilitas ekonomi', 'mobility', 'economic equality', 'kesetaraan ekonomi', 'equality', 'inclusive economy', 'ekonomi inklusif', 'inclusive',
        'inclusive growth', 'pertumbuhan inklusif', 'growth', 'economic transformation', 'transformasi ekonomi', 'transformation', 'economic progress', 'kemajuan ekonomi', 'progress',
        'economic reform', 'reformasi ekonomi', 'reform', 'economic innovation', 'inovasi ekonomi', 'innovation', 'economic diversity', 'diversifikasi ekonomi', 'diversity',
        'economic integration', 'integrasi ekonomi', 'integration', 'economic stability', 'economic resilience', 'economic recovery', 'economic recovery', 'economic outlook', 'economic outlook',
        'economic prospects', 'prospek ekonomi', 'economic policy', 'economic policy', 'economic strategy', 'economic strategy', 'economic dynamics', 'dinamika ekonomi', 'dynamics'
    ],
    'Control': [
        'kontrol', 'kendali', 'monitor', 'pengawasan', 'pengendalian', 'control', 'monitoring', 'supervision', 'oversight', 'pengawasan',
        'control system', 'sistem kontrol', 'control mechanisms', 'mekanisme kontrol', 'control processes', 'proses kontrol', 'quality control',
        'kontrol kualitas', 'quality assurance', 'jaminan kualitas', 'quality management', 'manajemen kualitas', 'control measures', 'langkah-langkah kontrol',
        'measures', 'control standards', 'standar kontrol', 'standards', 'control objectives', 'tujuan kontrol', 'objectives', 'control framework', 'kerangka kerja kontrol',
        'framework', 'control environment', 'lingkungan kontrol', 'environment', 'control activities', 'aktivitas kontrol', 'activities', 'control techniques', 'teknik kontrol',
        'techniques', 'control tools', 'alat kontrol', 'tools', 'control strategies', 'strategi kontrol', 'strategies', 'risk control', 'kontrol risiko', 'risk management',
        'manajemen risiko', 'risk mitigation', 'mitigasi risiko', 'mitigation', 'control systems', 'sistem kontrol', 'system', 'control procedures', 'prosedur kontrol',
        'procedures', 'control tests', 'uji kontrol', 'tests', 'internal control', 'kontrol internal', 'internal audit', 'audit internal', 'audit', 'internal review',
        'tinjauan internal', 'review', 'compliance', 'kepatuhan', 'compliance control', 'kontrol kepatuhan', 'control of compliance', 'kepatuhan kontrol', 'compliance measures',
        'langkah-langkah kepatuhan', 'measures', 'compliance monitoring', 'pemantauan kepatuhan', 'monitoring', 'regulatory control', 'kontrol regulasi', 'regulation',
        'regulasi', 'control of regulations', 'kontrol terhadap regulasi', 'control of standards', 'kontrol terhadap standar', 'standards', 'control management',
        'manajemen kontrol', 'management', 'control assessment', 'penilaian kontrol', 'assessment', 'control evaluation', 'evaluasi kontrol', 'evaluation', 'control policies',
        'kebijakan kontrol', 'policies', 'control procedures', 'prosedur kontrol', 'procedures', 'financial control', 'kontrol keuangan', 'financial', 'keuangan', 'control of financial',
        'kontrol keuangan', 'financial monitoring', 'pemantauan keuangan', 'monitoring', 'control of budget', 'kontrol anggaran', 'budget', 'control budget', 'kontrol terhadap anggaran',
        'budget control', 'anggaran kontrol', 'control plan', 'rencana kontrol', 'plan', 'control planning', 'perencanaan kontrol', 'planning', 'control system', 'sistem kontrol', 'control framework',
        'kerangka kontrol', 'framework', 'control mechanisms', 'mekanisme kontrol', 'mechanisms', 'control processes', 'proses kontrol', 'processes', 'control and supervision', 'kontrol dan pengawasan',
        'supervision', 'control of operations', 'kontrol operasi', 'operations', 'operational control', 'kontrol operasional', 'operational', 'control performance', 'kinerja kontrol', 'performance', 'control of activities',
        'kontrol kegiatan', 'activities', 'control of tasks', 'kontrol tugas', 'tasks', 'control over work', 'kontrol pekerjaan', 'work', 'work control', 'kontrol pekerjaan', 'job control', 'kontrol pekerjaan',
        'control job', 'kontrol pekerjaan', 'work supervision', 'pengawasan kerja', 'supervision', 'work monitoring', 'pemantauan kerja', 'monitoring', 'control mechanisms', 'mekanisme kontrol', 'mechanisms', 'control system',
        'sistem kontrol', 'system', 'control environment', 'lingkungan kontrol', 'environment', 'control assessment', 'penilaian kontrol', 'assessment', 'control evaluation', 'evaluasi kontrol', 'evaluation', 'internal control',
        'kontrol internal', 'internal', 'control measures', 'langkah-langkah kontrol', 'measures', 'risk control', 'kontrol risiko', 'risk', 'control of compliance', 'kontrol kepatuhan', 'compliance', 'regulatory control', 'kontrol regulasi',
        'regulatory', 'financial control', 'kontrol keuangan', 'financial', 'control procedures', 'prosedur kontrol', 'procedures', 'control policies', 'kebijakan kontrol', 'policies', 'control standards', 'standar kontrol', 'standards',
        'control strategies', 'strategi kontrol', 'strategies', 'control tools', 'alat kontrol', 'tools', 'control techniques', 'teknik kontrol', 'techniques', 'control framework', 'kerangka kontrol', 'framework', 'control processes', 'proses kontrol',
        'processes', 'control and supervision', 'kontrol dan pengawasan', 'supervision', 'operational control', 'kontrol operasional', 'operational', 'financial monitoring', 'pemantauan keuangan', 'monitoring', 'internal review', 'tinjauan internal', 'review'
    ],
    'Efficiency': [
        'efisien', 'efficient', 'efektif', 'hemat', 'produktif', 'efis', 'hemat waktu', 'efisien tinggi', 'waktu penggunaan yang efisien', 
        'penggunaan sumber daya yang hemat', 'efisiennya', 'tepat waktu hasil', 'hasil yang tepat waktu', 'efficiency', 'efisiensi', 'efisiensi biaya', 
        'cost efficiency', 'efisiensi energi', 'energy efficiency', 'efisiensi sumber daya', 'resource efficiency', 'efisiensi tenaga kerja', 
        'labor efficiency', 'efisiensi proses', 'process efficiency', 'efisiensi operasional', 'operational efficiency', 'efisiensi kerja', 
        'work efficiency', 'efisiensi kinerja', 'performance efficiency', 'efisiensi manajemen', 'management efficiency', 'efisiensi organisasi', 
        'organizational efficiency', 'efisiensi finansial', 'financial efficiency', 'efisiensi produksi', 'production efficiency', 'efisiensi distribusi', 
        'distribution efficiency', 'efisiensi logistik', 'logistics efficiency', 'efisiensi sistem', 'system efficiency', 'efisiensi teknologi', 
        'technology efficiency', 'efisiensi komunikasi', 'communication efficiency', 'efisiensi waktu', 'time efficiency', 'efisiensi biaya', 'cost efficiency', 
        'efisiensi pengeluaran', 'expenditure efficiency', 'efisiensi anggaran', 'budget efficiency', 'efisiensi ruang', 'space efficiency', 'efisiensi bahan baku', 
        'raw material efficiency', 'efisiensi energi', 'energy efficiency', 'efisiensi bahan bakar', 'fuel efficiency', 'efisiensi listrik', 'electricity efficiency', 
        'efisiensi air', 'water efficiency', 'efisiensi gas', 'gas efficiency', 'efisiensi bahan', 'material efficiency', 'efisiensi kimia', 'chemical efficiency', 
        'efisiensi pabrik', 'factory efficiency', 'efisiensi mesin', 'machine efficiency', 'efisiensi peralatan', 'equipment efficiency', 'efisiensi kendaraan', 
        'vehicle efficiency', 'efisiensi transportasi', 'transportation efficiency', 'efisiensi operasional', 'operational efficiency', 'efisiensi produksi', 
        'production efficiency', 'efisiensi manufaktur', 'manufacturing efficiency', 'efisiensi perakitan', 'assembly efficiency', 'efisiensi kualitas', 
        'quality efficiency', 'efisiensi kerja', 'work efficiency', 'efisiensi produktivitas', 'productivity efficiency', 'efisiensi kinerja', 
        'performance efficiency', 'efisiensi pengembangan', 'development efficiency', 'efisiensi penelitian', 'research efficiency', 'efisiensi pengujian', 
        'testing efficiency', 'efisiensi pelatihan', 'training efficiency', 'efisiensi keuangan', 'financial efficiency', 'efisiensi akuntansi', 
        'accounting efficiency', 'efisiensi perbankan', 'banking efficiency', 'efisiensi pembayaran', 'payment efficiency', 'efisiensi penerimaan', 
        'revenue efficiency', 'efisiensi penjualan', 'sales efficiency', 'efisiensi pemasaran', 'marketing efficiency', 'efisiensi promosi', 'promotion efficiency', 
        'efisiensi iklan', 'advertising efficiency', 'efisiensi hubungan pelanggan', 'customer relations efficiency', 'efisiensi layanan pelanggan', 
        'customer service efficiency', 'efisiensi kepuasan pelanggan', 'customer satisfaction efficiency', 'efisiensi komunikasi', 'communication efficiency', 
        'efisiensi informasi', 'information efficiency', 'efisiensi data', 'data efficiency', 'efisiensi IT', 'IT efficiency', 'efisiensi sistem informasi', 
        'information systems efficiency', 'efisiensi perangkat lunak', 'software efficiency', 'efisiensi perangkat keras', 'hardware efficiency', 
        'efisiensi jaringan', 'network efficiency', 'efisiensi internet', 'internet efficiency', 'efisiensi media sosial', 'social media efficiency', 
        'efisiensi email', 'email efficiency', 'efisiensi keamanan', 'security efficiency', 'efisiensi privasi', 'privacy efficiency', 'efisiensi pengelolaan', 
        'management efficiency', 'efisiensi administrasi', 'administrative efficiency', 'efisiensi pelayanan', 'service efficiency', 'efisiensi kebijakan', 
        'policy efficiency', 'efisiensi regulasi', 'regulatory efficiency', 'efisiensi prosedur', 'procedural efficiency', 'efisiensi proses bisnis', 
        'business process efficiency', 'efisiensi operasional', 'operational efficiency', 'efisiensi manajemen proyek', 'project management efficiency', 
        'efisiensi sumber daya manusia', 'human resource efficiency', 'efisiensi SDM', 'HR efficiency', 'efisiensi rekrutmen', 'recruitment efficiency', 
        'efisiensi pengembangan SDM', 'HR development efficiency', 'efisiensi pelatihan', 'training efficiency', 'efisiensi kerja tim', 'team efficiency', 
        'efisiensi kolaborasi', 'collaboration efficiency', 'efisiensi interaksi', 'interaction efficiency', 'efisiensi koordinasi', 'coordination efficiency', 
        'efisiensi kepemimpinan', 'leadership efficiency', 'efisiensi pengambilan keputusan', 'decision-making efficiency', 'efisiensi strategi', 'strategy efficiency', 
        'efisiensi perencanaan', 'planning efficiency', 'efisiensi analisis', 'analysis efficiency', 'efisiensi evaluasi', 'evaluation efficiency', 'efisiensi kontrol', 
        'control efficiency', 'efisiensi penilaian', 'assessment efficiency', 'efisiensi pengawasan', 'supervision efficiency', 'efisiensi audit', 'audit efficiency', 
        'efisiensi pemantauan', 'monitoring efficiency', 'efisiensi penjaminan kualitas', 'quality assurance efficiency', 'efisiensi manajemen risiko', 'risk management efficiency', 
        'efisiensi keberlanjutan', 'sustainability efficiency', 'efisiensi lingkungan', 'environmental efficiency', 'efisiensi sosial', 'social efficiency', 'efisiensi budaya', 
        'cultural efficiency', 'efisiensi teknologi', 'technological efficiency', 'efisiensi inovasi', 'innovation efficiency', 'efisiensi kreativitas', 'creativity efficiency', 
        'efisiensi pengetahuan', 'knowledge efficiency', 'efisiensi pembelajaran', 'learning efficiency', 'efisiensi pendidikan', 'educational efficiency', 'efisiensi sekolah', 
        'school efficiency', 'efisiensi universitas', 'university efficiency', 'efisiensi akademik', 'academic efficiency', 'efisiensi penelitian', 'research efficiency', 
        'efisiensi pengembangan', 'development efficiency', 'efisiensi pertumbuhan', 'growth efficiency', 'efisiensi perubahan', 'change efficiency', 'efisiensi transformasi', 
        'transformation efficiency', 'efisiensi adaptasi', 'adaptation efficiency', 'efisiensi mitigasi', 'mitigation efficiency', 'efisiensi pemulihan', 'recovery efficiency', 
        'efisiensi tanggap darurat', 'emergency response efficiency', 'efisiensi kesehatan', 'health efficiency', 'efisiensi medis', 'medical efficiency', 'efisiensi rumah sakit', 
        'hospital efficiency', 'efisiensi pelayanan kesehatan', 'health service efficiency', 'efisiensi perawatan', 'care efficiency', 'efisiensi perawatan pasien', 
        'patient care efficiency', 'efisiensi farmasi', 'pharmacy efficiency', 'efisiensi obat', 'drug efficiency', 'efisiensi terapi', 'therapy efficiency', 'efisiensi klinis', 
        'clinical efficiency', 'efisiensi gizi', 'nutrition efficiency', 'efisiensi kebugaran', 'fitness efficiency', 'efisiensi olahraga', 'sports efficiency', 
        'efisiensi latihan', 'exercise efficiency', 'efisiensi diet', 'diet efficiency', 'efisiensi gaya hidup', 'lifestyle efficiency', 'efisiensi transportasi', 
        'transportation efficiency', 'efisiensi perjalanan', 'travel efficiency', 'efisiensi mobilitas', 'mobility efficiency', 'efisiensi kendaraan', 'vehicle efficiency', 
        'efisiensi energi', 'energy efficiency', 'efisiensi bahan bakar', 'fuel efficiency', 'efisiensi energi terbarukan', 'renewable energy efficiency', 'efisiensi listrik', 
        'electricity efficiency', 'efisiensi air', 'water efficiency', 'efisiensi gas', 'gas efficiency', 'efisiensi bahan', 'material efficiency', 'efisiensi kimia', 'chemical efficiency', 
        'efisiensi pabrik', 'factory efficiency', 'efisiensi mesin', 'machine efficiency', 'efisiensi peralatan', 'equipment efficiency'
    ],
    'Service': [
        'layanan', 'pelayanan', 'bantuan', 'service', 'customer service', 'customer support', 'dukungan pelanggan', 'support', 'bantuan teknis', 'technical support',
        'servis', 'perbaikan', 'repair', 'maintenance', 'pemeliharaan', 'help', 'bantuan', 'helpdesk', 'meja bantuan', 'service center', 'pusat layanan', 
        'service quality', 'kualitas layanan', 'quality', 'customer care', 'perawatan pelanggan', 'care', 'klien', 'client', 'client support', 'dukungan klien', 
        'customer satisfaction', 'kepuasan pelanggan', 'satisfaction', 'response time', 'waktu respon', 'respon', 'response', 'kecepatan respon', 'response speed', 
        'fast response', 'respon cepat', 'response rate', 'tingkat respon', 'response efficiency', 'efisiensi respon', 'efisiensi layanan', 'service efficiency', 
        'service effectiveness', 'efektivitas layanan', 'effective service', 'layanan efektif', 'responsive service', 'layanan responsif', 'responsive', 'responsif', 
        'proactive service', 'layanan proaktif', 'proactive', 'proaktif', 'customer feedback', 'umpan balik pelanggan', 'feedback', 'customer reviews', 'ulasan pelanggan', 
        'reviews', 'complaints', 'keluhan', 'complaint handling', 'penanganan keluhan', 'complaint resolution', 'penyelesaian keluhan', 'resolution', 'resolusi', 
        'customer issues', 'masalah pelanggan', 'issues', 'masalah', 'problem solving', 'pemecahan masalah', 'solving', 'resolution', 'resolusi', 'customer engagement', 
        'keterlibatan pelanggan', 'engagement', 'loyalty', 'loyalitas', 'customer loyalty', 'loyalitas pelanggan', 'loyalty programs', 'program loyalitas', 'customer retention', 
        'retensi pelanggan', 'retention', 'service availability', 'ketersediaan layanan', 'availability', 'availability of service', 'service accessibility', 'aksesibilitas layanan', 
        'accessibility', 'service reliability', 'keandalan layanan', 'reliability', 'reliable service', 'layanan andal', 'trustworthy service', 'layanan terpercaya', 'trust', 
        'kepercayaan', 'trustworthiness', 'terpercaya', 'customer trust', 'kepercayaan pelanggan', 'service transparency', 'transparansi layanan', 'transparency', 'transparan', 
        'service communication', 'komunikasi layanan', 'communication', 'komunikasi', 'effective communication', 'komunikasi efektif', 'communication channels', 'saluran komunikasi', 
        'channels', 'service channels', 'saluran layanan', 'service delivery', 'pengiriman layanan', 'delivery', 'service follow-up', 'tindak lanjut layanan', 'follow-up', 
        'follow-up service', 'tindak lanjut', 'service response', 'respon layanan', 'response', 'after-sales service', 'layanan purna jual', 'after-sales', 'purna jual', 
        'service network', 'jaringan layanan', 'network', 'network support', 'dukungan jaringan', 'network service', 'layanan jaringan', 'digital service', 'layanan digital', 
        'digital', 'online service', 'layanan online', 'online', 'self-service', 'layanan mandiri', 'self', 'automated service', 'layanan otomatis', 'automated', 'otomatis', 
        'service automation', 'otomatisasi layanan', 'automation', 'service innovation', 'inovasi layanan', 'innovation', 'innovative service', 'layanan inovatif', 'innovative', 
        'service development', 'pengembangan layanan', 'development', 'service improvement', 'peningkatan layanan', 'improvement', 'continuous improvement', 'peningkatan berkelanjutan', 
        'continuous', 'berkelanjutan', 'continuous service improvement', 'peningkatan layanan berkelanjutan', 'service management', 'manajemen layanan', 'management', 'service leadership', 
        'kepemimpinan layanan', 'leadership', 'service strategy', 'strategi layanan', 'strategy', 'service planning', 'perencanaan layanan', 'planning', 'service policies', 'kebijakan layanan', 
        'policies', 'service procedures', 'prosedur layanan', 'procedures', 'service standards', 'standar layanan', 'standards', 'customer-friendly service', 'layanan ramah pelanggan', 
        'customer-friendly', 'friendly service', 'ramah layanan', 'empathetic service', 'layanan empatik', 'empathetic', 'empathy', 'empati', 'personalized service', 'layanan personalisasi', 
        'personalized', 'customized service', 'layanan disesuaikan', 'customized', 'customization', 'personalization', 'personalisasi', 'individual service', 'layanan individu', 'individual', 
        'service flexibility', 'fleksibilitas layanan', 'flexibility', 'flexible service', 'layanan fleksibel', 'adaptable service', 'layanan yang dapat disesuaikan', 'adaptable', 'service adaptability', 
        'adaptabilitas layanan', 'adaptability', 'service excellence', 'keunggulan layanan', 'excellence', 'excellent service', 'layanan yang sangat baik', 'excellent', 'superior service', 
        'layanan unggul', 'superior', 'superior customer service', 'layanan pelanggan unggul', 'service performance', 'kinerja layanan', 'performance', 'service performance metrics', 
        'metrik kinerja layanan', 'metrics', 'service performance indicators', 'indikator kinerja layanan', 'indicators', 'performance indicators', 'indikator kinerja', 'service goals', 
        'tujuan layanan', 'goals', 'customer service goals', 'tujuan layanan pelanggan', 'service objectives', 'tujuan layanan', 'objectives', 'customer service objectives', 'tujuan layanan pelanggan'
    ]
}

# Fungsi untuk mengambil review dari Google Play Store
def fetch_reviews(app_id, count, country='id', lang='id'):

    try:
        apps_review, _ = reviews(
            app_id,
            country=country,
            lang=lang,
            count=count,
            filter_score_with=None,
            sort=Sort.NEWEST
        )

        st.write("Raw review:", len(apps_review))
        st.write(apps_review[:1])

        df = pd.DataFrame(apps_review)

        st.write("Jumlah review:", len(df))
        st.write("Kolom:", df.columns.tolist())
        st.write(df.head())

        return df

    except Exception as e:
        st.error(f"Google Play Scraper Error: {e}")
        return pd.DataFrame()
    
# Fungsi case folding
def case_folding(text):
    return text.lower()

# Fungsi tokenisasi
def tokenize(text):
    words = re.findall(r'\b\w+\b', text)
    return words

# Fungsi stemming menggunakan Sastrawi
def stem_text(text):
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    return stemmer.stem(text)

# Fungsi menghapus stopword menggunakan Sastrawi
def remove_stopwords(text):
    stopword_remover_factory = StopWordRemoverFactory()
    stopword_remover = stopword_remover_factory.create_stop_word_remover()
    return stopword_remover.remove(text)

# Fungsi untuk memproses teks secara lengkap
def process_text(text):
    text_lower = case_folding(text)
    tokens = tokenize(text_lower)
    stemmed_text = stem_text(' '.join(tokens))
    text_without_stopwords = remove_stopwords(stemmed_text)
    return text_without_stopwords

# Fungsi untuk memproses kolom pada DataFrame
def process_column(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].apply(process_text)

# Fungsi untuk analisis sentimen dengan VADER
def sentiment_analysis_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_dict = analyzer.polarity_scores(text)
    return sentiment_dict['compound']

# Fungsi untuk analisis sentimen dengan TextBlob
def sentiment_analysis_textblob(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Fungsi untuk mengubah nilai sentimen menjadi skala Likert
def sentiment_to_likert(score, scale=5):
    if scale == 5:
        if score >= 0.5:
            return 5  # Sangat Puas Sekali
        elif score >= 0.2:
            return 4  # Sangat Puas
        elif score >= -0.2:
            return 3  # Puas
        elif score >= -0.5:
            return 2  # Cukup Puas
        else:
            return 1  # Kurang Puas

# Fungsi untuk menemukan review yang mirip dengan kata kunci
def find_similar_reviews(reviews, keywords, threshold=0.05):
    similar_reviews = []
    for i, review in enumerate(reviews, start=1):
        similarity_score = 0
        for keyword in keywords:
            if keyword in review:
                similarity_score += 1
                break
        if len(review.split()) > 0 and similarity_score / len(review.split()) >= threshold:
            similar_reviews.append((i, review, similarity_score / len(review.split())))
    return similar_reviews

# Fungsi utama untuk menjalankan seluruh sistem
def main(app_id, count, sentiment_method='VADER', domain=None):

    df = fetch_reviews(app_id, count)

    st.write("Jumlah data:", len(df))
    st.write("Kolom:", df.columns.tolist())
    st.write(df.head())

    # pastikan data ada
    if df.empty:
        st.error("Data review kosong. Cek App ID atau koneksi Google Play Scraper.")
        return {}

    if 'content' not in df.columns:
        st.error(f"Kolom content tidak ditemukan. Kolom tersedia: {df.columns.tolist()}")
        return {}

    # preprocessing
    process_column(df, 'content')

    # sentiment
    if sentiment_method == 'VADER':
        df['polarity'] = df['content'].apply(sentiment_analysis_vader)

    elif sentiment_method == 'TextBlob':
        df['polarity'] = df['content'].apply(sentiment_analysis_textblob)

    # likert
    df['sentiment'] = df['polarity'].apply(lambda x: sentiment_to_likert(x, 5))

    results = {}

    if domain == 'All':

        for domain_name, keywords in keywords_similarity.items():

            similar_reviews = find_similar_reviews(
                df['content'], 
                keywords, 
                threshold=0.09
            )

            similar_reviews_df = pd.DataFrame(
                similar_reviews,
                columns=['index', 'review', 'similarity_score']
            )

            similar_reviews_df['userName'] = df.loc[
                similar_reviews_df['index'] - 1,
                'userName'
            ].values

            similar_reviews_df['score'] = df.loc[
                similar_reviews_df['index'] - 1,
                'score'
            ].values

            similar_reviews_df['sentiment'] = df.loc[
                similar_reviews_df['index'] - 1,
                'sentiment'
            ].values

            sentiment_mapping = {
                1: 'Kurang Puas',
                2: 'Cukup Puas',
                3: 'Puas',
                4: 'Sangat Puas',
                5: 'Sangat Puas Sekali'
            }

            similar_reviews_df['sentiment_label'] = (
                similar_reviews_df['sentiment']
                .map(sentiment_mapping)
            )

            results[domain_name] = similar_reviews_df

    else:

        keywords = keywords_similarity.get(domain, [])

        similar_reviews = find_similar_reviews(
            df['content'],
            keywords,
            threshold=0.09
        )

        similar_reviews_df = pd.DataFrame(
            similar_reviews,
            columns=['index', 'review', 'similarity_score']
        )

        similar_reviews_df['userName'] = df.loc[
            similar_reviews_df['index'] - 1,
            'userName'
        ].values

        similar_reviews_df['score'] = df.loc[
            similar_reviews_df['index'] - 1,
            'score'
        ].values

        similar_reviews_df['sentiment'] = df.loc[
            similar_reviews_df['index'] - 1,
            'sentiment'
        ].values

        sentiment_mapping = {
            1: 'Kurang Puas',
            2: 'Cukup Puas',
            3: 'Puas',
            4: 'Sangat Puas',
            5: 'Sangat Puas Sekali'
        }

        similar_reviews_df['sentiment_label'] = (
            similar_reviews_df['sentiment']
            .map(sentiment_mapping)
        )

        results[domain] = similar_reviews_df

    return results

# Streamlit interface
st.title("Google Play Store Review Analyzer")

app_id = st.text_input("Enter the App ID (e.g., id.or.muhammadiyah.quran):", "")
count = st.number_input("Enter the number of reviews to fetch:", min_value=1, value=1000, step=1)
sentiment_method = st.selectbox("Choose Sentiment Analysis Method:", ["VADER", "TextBlob"])
domain = st.selectbox("Choose Domain to Analyze:", ["Performance", "Information", "Economy", "Control", "Efficiency", "Service", "All"])

if st.button("Fetch and Analyze Reviews"):
    with st.spinner('Fetching reviews...'):
        results = main(app_id, count, sentiment_method, domain)
        
    for domain_name, similar_reviews_df in results.items():
        st.subheader(f"Results for {domain_name} Domain")
        st.dataframe(similar_reviews_df)

        csv = similar_reviews_df.to_csv(index=False)
        st.download_button(label=f"Download data as CSV ({domain_name})", data=csv, file_name=f'similar_reviews_{domain_name}.csv', mime='text/csv')

        # Visualize the sentiment distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='sentiment_label', data=similar_reviews_df, order=['Kurang Puas', 'Cukup Puas', 'Puas', 'Sangat Puas', 'Sangat Puas Sekali'], palette='viridis')
        plt.title(f'Distribusi Sentimen Ulasan yang Mirip ({domain_name})')
        plt.xlabel('Skala Likert Sentimen')
        plt.ylabel('Jumlah Ulasan')
        st.pyplot(plt)

        # Display total results
        total_results = len(similar_reviews_df)
        st.write(f"JUMLAH REVIEW YANG MASUK KE DALAM DOMAIN {domain_name.upper()} YAITU: {total_results}")

        # Calculate the average sentiment score
        total_sentiment = similar_reviews_df['sentiment'].sum()  # Menjumlahkan semua nilai sentiment
        jumlah_ulasan = similar_reviews_df['sentiment'].count()  # Menghitung jumlah ulasan
        rata_rata_sentiment = total_sentiment / jumlah_ulasan  # Membagi total nilai sentiment dengan jumlah ulasan
        st.write(f"RATA-RATA SENTIMEN: {rata_rata_sentiment:.2f}")

        # Calculate the count of each sentiment label
        sentiment_counts = similar_reviews_df['sentiment'].value_counts().sort_index()
        for label, count in sentiment_counts.items():
            st.write(f"Jumlah ulasan dengan sentiment label {label}: {count}")
