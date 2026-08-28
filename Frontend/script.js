// =========================
// ELEMENTS
// =========================

const uploadInput =
    document.getElementById("ctUpload");

const viewerImage =
    document.getElementById("viewerImage");

const prevBtn =
    document.getElementById("prevBtn");

const nextBtn =
    document.getElementById("nextBtn");

const sliceSlider =
    document.getElementById("sliceSlider");

const sliceNumber =
    document.getElementById("sliceNumber");

const generateReportBtn =
    document.getElementById("generateReport");

// =========================
// REPORT ELEMENTS
// =========================

const ageInput =
    document.getElementById("age");

const sexInput =
    document.getElementById("sex");

const smokingInput =
    document.getElementById("smoking");

const packYearsInput =
    document.getElementById("packYears");

const reportAge =
    document.getElementById("reportAge");

const reportSex =
    document.getElementById("reportSex");

const reportSmoking =
    document.getElementById("reportSmoking");

const reportPackYears =
    document.getElementById("reportPackYears");

// =========================
// IMAGE STORAGE
// =========================

let uploadedImages = [];

let currentSlice = 0;

// =========================
// DISPLAY IMAGE
// =========================

function displaySlice(index){

    if(uploadedImages.length === 0)
        return;

    viewerImage.src =
        uploadedImages[index];

    sliceNumber.textContent =
        index + 1;

    sliceSlider.value =
        index + 1;

}

// =========================
// FILE UPLOAD
// =========================

uploadInput.addEventListener(
    "change",
    function(event){

        const files =
            Array.from(event.target.files);

        uploadedImages = [];

        files.forEach(file => {

            const imageURL =
                URL.createObjectURL(file);

            uploadedImages.push(
                imageURL
            );

        });

        if(uploadedImages.length > 0){

            currentSlice = 0;

            displaySlice(
                currentSlice
            );

            sliceSlider.max =
                uploadedImages.length;

        }

    }
);

// =========================
// NEXT BUTTON
// =========================

nextBtn.addEventListener(
    "click",
    () => {

        if(
            currentSlice <
            uploadedImages.length - 1
        ){

            currentSlice++;

            displaySlice(
                currentSlice
            );

        }

    }
);

// =========================
// PREVIOUS BUTTON
// =========================

prevBtn.addEventListener(
    "click",
    () => {

        if(currentSlice > 0){

            currentSlice--;

            displaySlice(
                currentSlice
            );

        }

    }
);

// =========================
// SLIDER CONTROL
// =========================

sliceSlider.addEventListener(
    "input",
    () => {

        currentSlice =
            Number(
                sliceSlider.value
            ) - 1;

        displaySlice(
            currentSlice
        );

    }
);

// =========================
// GENERATE REPORT
// =========================

generateReportBtn.addEventListener(
    "click",
    () => {

        reportAge.textContent =
            ageInput.value || "-";

        reportSex.textContent =
            sexInput.value || "-";

        reportSmoking.textContent =
            smokingInput.value || "-";

        reportPackYears.textContent =
            packYearsInput.value || "-";

        alert(
            "Report Generated Successfully"
        );

    }
);

// =========================
// KEYBOARD NAVIGATION
// =========================

document.addEventListener(
    "keydown",
    event => {

        if(
            event.key === "ArrowRight"
        ){

            nextBtn.click();

        }

        if(
            event.key === "ArrowLeft"
        ){

            prevBtn.click();

        }

    }
);

// =========================
// INITIAL STATE
// =========================

viewerImage.src =
    "https://placehold.co/800x500/000000/FFFFFF?text=Upload+CT+Scan";

sliceSlider.max = 1;
sliceSlider.value = 1;
