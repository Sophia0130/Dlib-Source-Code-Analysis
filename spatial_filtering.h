    //// spatially_filter_image 进入
    //// 使用了SIMD加速
    
        template <
            typename in_image_type,
            typename out_image_type,
            typename EXP
            >
        rectangle float_spatially_filter_image (
            const in_image_type& in_img_,
            out_image_type& out_img_,
            const matrix_exp<EXP>& filter_,
            bool add_to
        )
        {
            const_temp_matrix<EXP> filter(filter_);
            DLIB_ASSERT(filter.size() != 0,
                "\trectangle spatially_filter_image()"
                << "\n\t You can't give an empty filter."
                << "\n\t filter.nr(): "<< filter.nr()
                << "\n\t filter.nc(): "<< filter.nc()
            );
            DLIB_ASSERT(is_same_object(in_img_, out_img_) == false,
                "\trectangle spatially_filter_image()"
                << "\n\tYou must give two different image objects"
            );


            const_image_view<in_image_type> in_img(in_img_);
            image_view<out_image_type> out_img(out_img_);

            // if there isn't any input image then don't do anything
            if (in_img.size() == 0)
            {
                out_img.clear();
                return rectangle();
            }

            out_img.set_size(in_img.nr(),in_img.nc());

            
            // figure out the range that we should apply the filter to
            const long first_row = filter.nr()/2;
            const long first_col = filter.nc()/2;
            const long last_row = in_img.nr() - ((filter.nr()-1)/2);
            const long last_col = in_img.nc() - ((filter.nc()-1)/2);

            const rectangle non_border = rectangle(first_col, first_row, last_col-1, last_row-1);
            if (!add_to)
                zero_border_pixels(out_img_, non_border); 

            //// 8个横向滤波器使用SIMD并行计算
            //// r、c对应特征的某一维使用检测窗口遍历的位置，m、n对应检测窗口内部卷积时的扫描
            // apply the filter to the image
            //for (long r = first_row+2; r < last_row-2; ++r)
            for (long r = first_row; r < last_row; ++r)
            {
                //long c = first_col + 2;
                //for (; c < last_col - 7; c += 9)
                long c = first_col;
                for (long c = first_col; c < last_col-7; c+=8)
                {
                    //// 每个滤波器在一个for循环里做3个横向特征的点积（注意：是对8个横向滤波器的并行计算，不是1个滤波器里3个横向特征的并行）
                    simd8f p,p2,p3;
                    simd8f temp = 0, temp2=0, temp3=0;
                    for (long m = 0; m < filter.nr(); ++m)
                    {
                        long n = 0;
                        for (; n < filter.nc()-2; n+=3)
                        {
                            // pull out the current pixel and put it into p
                            p.load(&in_img[r-first_row+m][c-first_col+n]);
                            p2.load(&in_img[r-first_row+m][c-first_col+n+1]);
                            p3.load(&in_img[r-first_row+m][c-first_col+n+2]);
                            temp += p*filter(m,n);
                            temp2 += p2*filter(m,n+1);
                            temp3 += p3*filter(m,n+2);
                        }
                        for (; n < filter.nc(); ++n)
                        {
                            // pull out the current pixel and put it into p
                            p.load(&in_img[r-first_row+m][c-first_col+n]);
                            temp += p*filter(m,n);
                        }
                    }
                    temp += temp2+temp3;

                    // save this pixel to the output image
                    if (add_to == false)
                    {
                        temp.store(&out_img[r][c]);
                    }
                    else
                    {
                        p.load(&out_img[r][c]);
                        temp += p;
                        temp.store(&out_img[r][c]);
                    }
                }   //8个横向
                //for (; c < last_col-2; ++c)
                for (; c < last_col; ++c)
                {
                    float p;
                    float temp = 0;
                    for (long m = 0; m < filter.nr(); ++m)
                    {
                        for (long n = 0; n < filter.nc(); ++n)
                        {
                            // pull out the current pixel and put it into p
                            p = in_img[r-first_row+m][c-first_col+n];
                            temp += p*filter(m,n);
                        }
                    }

                    // save this pixel to the output image
                    if (add_to == false)
                    {
                        out_img[r][c] = temp;
                    }
                    else
                    {
                        out_img[r][c] += temp;
                    }
                }
            }

            return non_border;
        }


    //// apply_filters_to_fhog 进入
    //// spatially_filter_image()
    //// 功能：对某一层HOG金字塔，31维中的某一维，使用对应的滤波器（检测窗口）遍历，得到该层的滤波结果

    template <
        typename in_image_type,
        typename out_image_type,
        typename EXP,
        typename T
        >
    typename enable_if_c<pixel_traits<typename image_traits<out_image_type>::pixel_type>::grayscale && 
                         is_float_filtering2<in_image_type,out_image_type,EXP>::value,rectangle>::type 
    spatially_filter_image (
        const in_image_type& in_img,
        out_image_type& out_img,
        const matrix_exp<EXP>& filter,
        T scale,
        bool use_abs = false,
        bool add_to = false
    )
    {
        if (use_abs == false)
        {
            if (scale == 1)
            {
                // spatially_filter_image 进入
                return impl::float_spatially_filter_image(in_img, out_img, filter, add_to);
            }
            else
            {
                return impl::float_spatially_filter_image(in_img, out_img, filter / scale, add_to);
            }
        }
        else
        {
            return impl::grayscale_spatially_filter_image(in_img, out_img, filter, scale, true, add_to);
        }
    }