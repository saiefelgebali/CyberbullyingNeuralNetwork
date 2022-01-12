using CyberbullyingAPI.Models;

namespace CyberbullyingAPI.Services
{
    public static class CyberbullyingService
    {
        private static readonly string ModelPath = "D:/Projects/ml_models/cyberbullying_model_best.json";
        private static readonly CyberbullyingModel Model;

        static CyberbullyingService()
        {
            Model = new CyberbullyingModel(ModelPath);
        }

        public static double Predict(string text) => Model.Predict(text);
    }
}
