using CyberbullyingAPI.Services;
using Microsoft.AspNetCore.Mvc;

namespace CyberbullyingAPI.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class CyberbullyingController : ControllerBase
    {
        static CyberbullyingController()
        {
        }

        [HttpGet()]
        public double Get(string text)
        {
            try
            {
                return CyberbullyingService.Predict(text);
            } catch
            {
                // If the text is not valid, return -1
                return -1;
            }
        }
    }
}
