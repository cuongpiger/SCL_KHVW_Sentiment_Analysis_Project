from django.http.response import JsonResponse
from django.shortcuts import render
from modules import shopee_model as ShopeeModel
from modules import model as Model
from modules import deep_model as DeepModel
import pandas as pd
import json
from django.views.decorators.csrf import csrf_exempt


judge = None

# Create your views here.
def home(request):
    return render(request, 'sentiment/home.html')


def get_predicted_sentiment(request):
    global judge

    if judge is None:
        emojier = Model.loadByPickle("models/rdforest_model_emojis.pickle")
        commenter = DeepModel.SentimentLSTM("models/lstm_model_comments_1/lstm_model_comments_1.h5", 
                                            "models/lstm_model_comments_1/lstm_tokenizer_comments_1.pickle") 

        judge = ShopeeModel.ShopeeSentiment(emojier, commenter)
        
        
    new_reviews = pd.DataFrame()
    new_reviews['review'] = [
        "áo đẹp. bố mình rất ưng và rất thíc nhé. lên mua nhé các bạn mình.😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝😝",
        "Áo khá ấm, vải mềm, không có chỉ thừa, tuy nhiên khi giặt thì có hơi nhiều phấn vải",
        "Áo nỉ khá dày, nhẹ tênh, giống hình, thời gian giao hàng nhanh!ok, còn đc tặng voucher xịn xò!!!!!!!",
        "Chất ổn nhưng dây quần như bị ố nhìn khá xấu. Về tổng quan thì ổn 👍",
        "Mỏng, chất dễ xù, ban đầu thấy chất khá cứng, nhưng form vẫn chấp nhận dc",
        "Giao màu khác trong ảnh. Nhìn già kinh luôn. Các b cẩn thận khi mua. Màu nó là màu nâu sẫm chứ k giống trong ảnh chút nào :))) Cho bố e, bố e còn k thèm thì đủ hiểu nó già ntn :)))",
        "Lười chẳng buồn chụp cái áo vì thất vọng toàn tập, áo màu chuẩn giống hình nhưng len nhão, rộng khủng khiếp, cổ trễ quá sâu, phần nách thì k hề ôm vào ng mà cứ bửa hết ra khả năng là may lỗi, sẽ k dám mua quần áo qua mạng nữa",
        "Áo màu k giống ảnh phần viền là màu nâu đậm thêm nữa cổ áo rất rộng 🙂🙂",
        "Áo khác trên ảnh nhiều lắm 🙂màu nhạt dã man xong cổ rộng tóm lại k đáng tiền",
        "Áo rộng thùng thình bố mình mặc còn vừa nên mình cho bố  luôn rồi. Màu áo ship về cũng không được đẹp cho lắm.Nói chung là hơi thất vọng."
    ]
        
    data = judge.predict(new_reviews['review']).to_numpy()
    response = []
    for row in data:
        response.append({
            'comment': row[0],
            'probability': f"{row[1][0]} - {row[1][1]}"
        })

    
    return JsonResponse({'response': response})

@csrf_exempt 
def ajax_predict_comment(request):
    global judge

    if judge is None:
        emojier = Model.loadByPickle("models/rdforest_model_emojis.pickle")
        commenter = DeepModel.SentimentLSTM("models/lstm_model_comments_1/lstm_model_comments_1.h5", 
                                            "models/lstm_model_comments_1/lstm_tokenizer_comments_1.pickle") 

        judge = ShopeeModel.ShopeeSentiment(emojier, commenter)
        
    
    if request.is_ajax():    
        new_reviews = pd.DataFrame()
        new_reviews['review'] = [
            "áo đẹp. bố mình rất ưng và rất thíc nhé. lên mua nhé các bạn mình.😝",
            str(request.POST.get('comment'))
        ]
        
        data = judge.predict(new_reviews['review']).to_numpy()
        response = [] 
        for row in data[1:]:
            response.append({
                'comment': row[0],
                'nega_prob': f"{row[1][0]}",
                'posi_prob': f"{row[1][1]}"
            })
    
        return JsonResponse({'negaProb': response[0]['nega_prob'], 'posiProb': response[0]['posi_prob']})