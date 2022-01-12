using CyberbullyingAPI.Models;
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
            return CyberbullyingService.Predict(text);
        }
    }
}
