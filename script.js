/*=========================================================
                LUNGINSIGHT AI DASHBOARD
                Part 1 - Navigation & UI
=========================================================*/

document.addEventListener("DOMContentLoaded", () => {

    /*==============================
            SIDEBAR NAVIGATION
    ==============================*/

    const sidebarItems = document.querySelectorAll(".sidebar li");
    const sections = document.querySelectorAll("section");

    sidebarItems.forEach((item, index) => {

        item.addEventListener("click", () => {

            sidebarItems.forEach(i => i.classList.remove("active"));
            item.classList.add("active");

            if (sections[index]) {

                sections[index].scrollIntoView({

                    behavior: "smooth",
                    block: "start"

                });

            }

        });

    });

    /*==============================
            TOPBAR EFFECT
    ==============================*/

    const topbar = document.querySelector(".topbar");

    window.addEventListener("scroll", () => {

        if (!topbar) return;

        if (window.scrollY > 40) {

            topbar.style.background = "rgba(10,18,35,0.92)";
            topbar.style.backdropFilter = "blur(16px)";
            topbar.style.boxShadow = "0 10px 25px rgba(0,0,0,.25)";

        } else {

            topbar.style.background = "transparent";
            topbar.style.backdropFilter = "blur(0px)";
            topbar.style.boxShadow = "none";

        }

    });

    /*==============================
        SCROLL TO TOP BUTTON
    ==============================*/

    const scrollBtn = document.createElement("button");

    scrollBtn.className = "scroll-top";
    scrollBtn.innerHTML = "↑";

    document.body.appendChild(scrollBtn);

    scrollBtn.addEventListener("click", () => {

        window.scrollTo({

            top: 0,
            behavior: "smooth"

        });

    });

    window.addEventListener("scroll", () => {

        if (window.scrollY > 500) {

            scrollBtn.classList.add("show");

        } else {

            scrollBtn.classList.remove("show");

        }

    });

});
/*=========================================================
            PART 2 - DASHBOARD ANIMATIONS
=========================================================*/

document.addEventListener("DOMContentLoaded", () => {

    /*==============================
        ANIMATED STATISTICS
    ==============================*/

    const statNumbers = document.querySelectorAll(".stat-card h2");

    const animateCounter = (element) => {

        const text = element.textContent.trim();

        const number = parseFloat(text.replace(/[^\d.]/g, ""));

        if (isNaN(number)) return;

        const suffix =
            text.includes("%") ? "%" :
            text.includes("+") ? "+" : "";

        let current = 0;
        const duration = 1500;
        const start = performance.now();

        function update(time){

            const progress = Math.min((time - start) / duration, 1);

            current = number * progress;

            if(suffix === "%"){

                element.textContent = current.toFixed(1) + "%";

            }else{

                element.textContent = Math.floor(current) + suffix;

            }

            if(progress < 1){

                requestAnimationFrame(update);

            }else{

                element.textContent = text;

            }

        }

        requestAnimationFrame(update);

    };

    const statObserver = new IntersectionObserver((entries)=>{

        entries.forEach(entry=>{

            if(entry.isIntersecting){

                const cards = entry.target.querySelectorAll("h2");

                cards.forEach(animateCounter);

                statObserver.unobserve(entry.target);

            }

        });

    },{

        threshold:0.4

    });

    const statsSection = document.querySelector(".stats");

    if(statsSection){

        statObserver.observe(statsSection);

    }

    /*==============================
        PROGRESS BAR ANIMATION
    ==============================*/

    const bars = document.querySelectorAll(".fill");

    const progressObserver = new IntersectionObserver((entries)=>{

        entries.forEach(entry=>{

            if(entry.isIntersecting){

                bars.forEach(bar=>{

                    const target = bar.style.width;

                    if(!target) return;

                    bar.style.width = "0%";

                    setTimeout(()=>{

                        bar.style.transition = "width 2s ease";

                        bar.style.width = target;

                    },200);

                });

                progressObserver.disconnect();

            }

        });

    },{

        threshold:0.4

    });

    const performanceSection = document.querySelector(".performance-grid");

    if(performanceSection){

        progressObserver.observe(performanceSection);

    }

    /*==============================
        FUSION SCORE
    ==============================*/

    const score = document.querySelector(".score-circle h1");

    if(score){

        const finalScore = parseInt(score.textContent);

        score.textContent = "0%";

        const fusionObserver = new IntersectionObserver((entries)=>{

            entries.forEach(entry=>{

                if(entry.isIntersecting){

                    let value = 0;

                    const timer = setInterval(()=>{

                        value++;

                        score.textContent = value + "%";

                        if(value >= finalScore){

                            clearInterval(timer);

                        }

                    },25);

                    fusionObserver.disconnect();

                }

            });

        },{

            threshold:0.5

        });

        fusionObserver.observe(score);

    }

});
/*=========================================================
            PART 3 - INTERACTIONS & ANIMATIONS
=========================================================*/

document.addEventListener("DOMContentLoaded", () => {

    /*==============================
        SCROLL REVEAL
    ==============================*/

    const revealElements = document.querySelectorAll(
        ".workflow-card, .dashboard-card, .performance-card, .explain-card, .report-card, .about-card, .fusion-container"
    );

    revealElements.forEach(element => {

        element.style.opacity = "0";
        element.style.transform = "translateY(40px)";
        element.style.transition = "opacity .8s ease, transform .8s ease";

    });

    const revealObserver = new IntersectionObserver((entries) => {

        entries.forEach(entry => {

            if (entry.isIntersecting) {

                entry.target.style.opacity = "1";
                entry.target.style.transform = "translateY(0)";

                revealObserver.unobserve(entry.target);

            }

        });

    }, {

        threshold: 0.2

    });

    revealElements.forEach(element => {

        revealObserver.observe(element);

    });

    /*==============================
        CARD HOVER EFFECT
    ==============================*/

    const cards = document.querySelectorAll(
        ".dashboard-card, .performance-card, .about-card"
    );

    cards.forEach(card => {

        card.addEventListener("mousemove", e => {

            const rect = card.getBoundingClientRect();

            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const rotateX = ((rect.height / 2 - y) / rect.height) * 8;
            const rotateY = ((x - rect.width / 2) / rect.width) * 8;

            card.style.transform = `
                perspective(1000px)
                rotateX(${rotateX}deg)
                rotateY(${rotateY}deg)
                scale(1.02)
            `;

        });

        card.addEventListener("mouseleave", () => {

            card.style.transform = "";

        });

    });

    /*==============================
        BUTTON RIPPLE EFFECT
    ==============================*/

    const buttons = document.querySelectorAll(".primary-btn, .secondary-btn");

    buttons.forEach(button => {

        button.addEventListener("click", function(e){

            const ripple = document.createElement("span");

            ripple.className = "ripple";

            const rect = this.getBoundingClientRect();

            ripple.style.left = (e.clientX - rect.left) + "px";
            ripple.style.top = (e.clientY - rect.top) + "px";

            this.appendChild(ripple);

            setTimeout(() => {

                ripple.remove();

            }, 600);

        });

    });

    /*==============================
        LIVE CLOCK
    ==============================*/

    const clockContainer = document.querySelector(".right-side");

    if(clockContainer){

        const clock = document.createElement("span");

        clock.className = "live-clock";

        clockContainer.prepend(clock);

        function updateClock(){

            const now = new Date();

            clock.textContent = now.toLocaleTimeString([],{

                hour:"2-digit",
                minute:"2-digit",
                second:"2-digit"

            });

        }

        updateClock();

        setInterval(updateClock,1000);

    }

});
/*=========================================================
            PART 4 - FINAL POLISH
=========================================================*/

document.addEventListener("DOMContentLoaded", () => {

    /*==============================
        FLOATING BACKGROUND DOTS
    ==============================*/

    for(let i = 0; i < 12; i++){

        const dot = document.createElement("div");

        dot.className = "floating-dot";

        dot.style.left = Math.random() * 100 + "%";
        dot.style.top = Math.random() * 100 + "%";
        dot.style.animationDelay = Math.random() * 8 + "s";
        dot.style.animationDuration = (8 + Math.random() * 6) + "s";

        document.body.appendChild(dot);

    }

    /*==============================
        STATUS PULSE
    ==============================*/

    const statusDots = document.querySelectorAll(".online");

    if(statusDots.length){

        setInterval(()=>{

            statusDots.forEach(dot=>{

                dot.classList.add("pulse");

                setTimeout(()=>{

                    dot.classList.remove("pulse");

                },500);

            });

        },1800);

    }

    /*==============================
        ACTIVE SIDEBAR
    ==============================*/

    const sidebarItems = document.querySelectorAll(".sidebar li");
    const sections = document.querySelectorAll("section");

    window.addEventListener("scroll",()=>{

        let current = 0;

        sections.forEach((section,index)=>{

            const top = section.offsetTop - 150;

            if(window.scrollY >= top){

                current = index;

            }

        });

        sidebarItems.forEach(item=>item.classList.remove("active"));

        if(sidebarItems[current]){

            sidebarItems[current].classList.add("active");

        }

    });

    /*==============================
        CONSOLE MESSAGE
    ==============================*/

    console.log(
        "%c✔ LungInsight AI Dashboard Loaded",
        "color:#3b82f6;font-size:16px;font-weight:bold;"
    );

});