using CyberbullyingAPI.Models;
using CyberbullyingAPI.Services;
using Microsoft.AspNetCore.Mvc;
using System.Net;
using System.Net.Http;
using System.Web;

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
                return -1;
            }
        }
    }
}
