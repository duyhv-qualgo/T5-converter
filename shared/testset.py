"""
Test set for T5 EN↔VI translation evaluation.

60 sentence pairs (30 EN→VI + 30 VI→EN) across 6 domains:
  greetings, travel, food, healthcare, work, general

Task token IDs:  EN→VI = 20000,  VI→EN = 20001
"""

TASK_EN_VI = 20000
TASK_VI_EN = 20001

EN_VI = [
    # Greetings / daily conversation
    ("Hello, how are you?",                                        "Xin chào, anh khoẻ không?"),
    ("Good morning! Did you sleep well?",                          "Chào buổi sáng! Anh có ngủ ngon không?"),
    ("It's nice to meet you.",                                     "Rất vui được gặp anh."),
    ("See you tomorrow.",                                          "Hẹn gặp lại anh vào ngày mai."),
    ("Thank you very much for your help.",                         "Cảm ơn anh rất nhiều vì đã giúp đỡ."),
    ("I'm sorry, I don't understand.",                             "Xin lỗi, tôi không hiểu."),
    ("Please speak more slowly.",                                  "Xin hãy nói chậm hơn."),
    ("What is your name?",                                         "Tên bạn là gì?"),
    ("My name is John and I work as a software engineer.",         "Tên tôi là John và tôi làm việc như một kỹ sư phần mềm."),
    ("She has been studying Vietnamese for three years.",          "Cô đã học tiếng Việt trong ba năm."),
    # Travel / directions
    ("Can you help me find the nearest train station?",            "Anh có thể giúp tôi tìm đài ở gần nhất không?"),
    ("How far is it to the airport?",                              "Sân bay cách đây bao xa?"),
    ("Please take me to this address.",                            "Xin hãy đưa tôi đến địa chỉ này."),
    ("The bus leaves at seven o'clock in the morning.",            "Xe buýt khởi hành lúc bảy giờ sáng."),
    ("Is there a taxi available near here?",                       "Có taxi nào ở gần đây không?"),
    ("I would like to book a hotel room for two nights.",          "Tôi muốn đặt phòng khách sạn trong hai đêm."),
    ("Turn left at the traffic light.",                            "Rẽ trái tại đèn giao thông."),
    ("The museum is closed on Mondays.",                           "Bảo tàng đóng cửa vào thứ Hai."),
    ("How long does the flight take?",                             "Chuyến bay mất bao lâu?"),
    ("I lost my passport and need help.",                          "Tôi bị mất hộ chiếu và cần giúp đỡ."),
    # Food / restaurant
    ("I would like to order a coffee please.",                     "Tôi muốn gọi cà phê."),
    ("The weather is beautiful today.",                            "Thời tiết đẹp hôm nay."),
    ("Do you have any vegetarian dishes on the menu?",             "Thực đơn có món ăn chay không?"),
    ("This food is very delicious.",                               "Món ăn này rất ngon."),
    ("Can I have the bill please?",                                "Cho tôi xin hóa đơn."),
    ("I am allergic to peanuts.",                                  "Tôi bị dị ứng với đậu phộng."),
    ("The restaurant opens at six in the evening.",                "Nhà hàng mở cửa lúc sáu giờ tối."),
    ("I would like a table for four people.",                      "Tôi muốn một bàn cho bốn người."),
    ("Please bring me a glass of water.",                          "Xin hãy mang cho tôi một ly nước."),
    ("Is the soup spicy?",                                         "Canh có cay không?"),
]

VI_EN = [
    # Greetings / daily conversation
    ("Xin chào, bạn có khỏe không?",                              "Hello, are you okay?"),
    ("Hôm nay trời đẹp quá.",                                     "It's beautiful today."),
    ("Rất vui được gặp bạn.",                                     "It's nice to meet you."),
    ("Hẹn gặp lại bạn vào ngày mai.",                             "See you tomorrow."),
    ("Cảm ơn bạn rất nhiều.",                                     "Thank you very much."),
    ("Xin lỗi, tôi không hiểu.",                                  "I'm sorry, I don't understand."),
    ("Tên tôi là Lan và tôi là giáo viên.",                       "My name is Lan and I am a teacher."),
    ("Anh ấy đang học tiếng Anh.",                                "He is studying English."),
    ("Chúng tôi sống ở Hà Nội.",                                  "We live in Hanoi."),
    ("Tôi thích đọc sách vào buổi tối.",                          "I like reading books in the evening."),
    # Travel / directions
    ("Bạn có thể giúp tôi tìm ga tàu gần nhất không?",           "Can you help me find the nearest station?"),
    ("Sân bay cách đây bao xa?",                                  "How far is the airport from here?"),
    ("Xe buýt khởi hành lúc mấy giờ?",                           "What time does the bus leave?"),
    ("Tôi muốn đặt phòng khách sạn.",                             "I want to book a hotel room."),
    ("Hãy rẽ trái tại đèn giao thông.",                           "Turn left at the traffic light."),
    ("Tôi bị lạc và cần giúp đỡ.",                                "I am lost and need help."),
    ("Chuyến bay đến Hà Nội mất bao lâu?",                       "How long does the flight to Hanoi take?"),
    ("Có taxi nào ở đây không?",                                  "Is there a taxi here?"),
    ("Bảo tàng mở cửa lúc mấy giờ?",                             "What time does the museum open?"),
    ("Xin hãy đưa tôi đến địa chỉ này.",                         "Please take me to this address."),
    # Food / restaurant
    ("Tôi muốn đặt một tách cà phê.",                             "I want to put a cup of coffee."),
    ("Thực đơn có món chay không?",                               "Is there a vegetarian dish on the menu?"),
    ("Món này rất ngon.",                                         "This dish is very delicious."),
    ("Cho tôi xin hóa đơn.",                                     "Please give me the bill."),
    ("Tôi bị dị ứng với hải sản.",                               "I am allergic to seafood."),
    ("Nhà hàng mở cửa lúc sáu giờ tối.",                         "The restaurant opens at six in the evening."),
    ("Xin hãy mang cho tôi một ly nước.",                         "Please bring me a glass of water."),
    # Healthcare / medical
    ("Bệnh viện nằm trên đường chính gần công viên.",             "The hospital is on the main road near the park."),
    ("Tôi cần gặp bác sĩ ngay.",                                  "I need to see a doctor immediately."),
    ("Đầu tôi bị đau từ sáng đến giờ.",                          "My head has been hurting since this morning."),
]

# Flat list: (task_id, source, reference)
ALL_PAIRS = (
    [(TASK_EN_VI, src, ref) for src, ref in EN_VI] +
    [(TASK_VI_EN, src, ref) for src, ref in VI_EN]
)


def load_phomt(n_samples: int = 200, split: str = "test",
               seed: int = 42) -> list:
    """
    Load sentence pairs from ura-hcmut/PhoMT (HuggingFace).

    Returns a flat list of (task_id, source, reference) tuples with
    n_samples // 2 EN→VI pairs and n_samples // 2 VI→EN pairs drawn
    from the requested split.

    Args:
        n_samples: total pairs to return (split evenly between directions).
        split:     HuggingFace dataset split — "test", "validation", or "train".
        seed:      random seed for reproducible sampling.
    """
    from datasets import load_dataset
    import random

    ds = load_dataset("ura-hcmut/PhoMT", split=split)

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    half = n_samples // 2
    selected = indices[:half * 2]

    en_vi_idx = selected[:half]
    vi_en_idx = selected[half:]

    pairs = []
    for i in en_vi_idx:
        row = ds[i]
        pairs.append((TASK_EN_VI, row["en"], row["vi"]))
    for i in vi_en_idx:
        row = ds[i]
        pairs.append((TASK_VI_EN, row["vi"], row["en"]))

    return pairs
