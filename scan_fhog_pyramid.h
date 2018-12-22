		//// apply_filters_to_fhog()函数
    	//// 功能：对HOG特征金字塔的某一层滤波（31维滤波后的结果相加）
    	//// 参数：- saliency_image 	 各维滤波相加后的结果，saliency_image的一个像素，对应于特征层中的一个检测窗口
		////	   -feats    某一层特征		- feats[i]	   31维中的某一维
    	//// 	   - area				 滤波后，未被0填充的矩形区域
    	
        template <typename fhog_filterbank>
        rectangle apply_filters_to_fhog (
            const fhog_filterbank& w,
            const array<array2d<float> >& feats,
            array2d<float>& saliency_image
        )
        {

			std::cout << "apply_filters_to_fhog()" << endl;

            const unsigned long num_separable_filters = w.num_separable_filters();	
            rectangle area;

			//// 对31维中的某一维做滤波卷积，并将结果相加
			
			//// 1.在该条件下使用 regular filter 
            // use the separable filters if they would be faster than running the regular filters.
            if (num_separable_filters > w.filters.size()*std::min(w.filters[0].nr(),w.filters[0].nc())/3.0)		//计算复杂度的公式得出
            {
                area = spatially_filter_image(feats[0], saliency_image, w.filters[0]);					//第一次滤波结果重写
                for (unsigned long i = 1; i < w.filters.size(); ++i)
                {
                	//// 对HOG金字塔的某一层某一维的，即31维中的某一维做滤波卷积
                    // now we filter but the output adds to saliency_image rather than
                    // overwriting it.
                    spatially_filter_image(feats[i], saliency_image, w.filters[i], 1, false, true);		//将31维的滤波结果相加
                    
                }

				std::cout << "Use Regular Filter" << endl;
				std::cout << "filters-size" << w.filters.size() << endl; //31维
            }

			//// 2.在该条件下使用seperable filter可分离滤波器
            else
            {
                saliency_image.clear();
                array2d<float> scratch;

                // find the first filter to apply
                unsigned long i = 0;
                while (i < w.row_filters.size() && w.row_filters[i].size() == 0) 
                    ++i;

				//// 分成x、y方向两个独立的一维滤波器
                for (; i < w.row_filters.size(); ++i)
                {
                    for (unsigned long j = 0; j < w.row_filters[i].size(); ++j)
                    {
                        if (saliency_image.size() == 0)
                            area = float_spatially_filter_image_separable(feats[i], saliency_image, w.row_filters[i][j], w.col_filters[i][j],scratch,false);
                        else
                            area = float_spatially_filter_image_separable(feats[i], saliency_image, w.row_filters[i][j], w.col_filters[i][j],scratch,true);
                    }
                }
                if (saliency_image.size() == 0)
                {
                    saliency_image.set_size(feats[0].nr(), feats[0].nc());
                    assign_all_pixels(saliency_image, 0);
                }

				std::cout << "Use Separable Filter" << endl;
				std::cout << "row col filter" << w.row_filters.size() << endl;

            }
            return area;
        }
    }


        //// 功能：创建fhog金字塔
        template <
            typename pyramid_type,
            typename image_type,
            typename feature_extractor_type
        >
        void create_fhog_pyramid(
        const image_type& img,
        const feature_extractor_type& fe,
        array<array<array2d<float> > >& feats,
        int cell_size,
        int filter_rows_padding,
        int filter_cols_padding,
        unsigned long min_pyramid_layer_width,
        unsigned long min_pyramid_layer_height,
        unsigned long max_pyramid_levels
        )
        {
            unsigned long levels = 0;
            rectangle rect = get_rect(img);

            // 根据最小金字塔层的尺寸，确定金字塔的层数
            // figure out how many pyramid levels we should be using based on the image size

            pyramid_type pyr;

            do
            {
                rect = pyr.rect_down(rect);
                ++levels;
            } while (rect.width() >= min_pyramid_layer_width && rect.height() >= min_pyramid_layer_height &&
                levels < max_pyramid_levels);

            if (feats.max_size() < levels)
                feats.set_max_size(levels);
            feats.set_size(levels);

            // build our feature pyramid
            fe(img, feats[0], cell_size, filter_rows_padding, filter_cols_padding); //第一层HOG金字塔

            DLIB_ASSERT(feats[0].size() == fe.get_num_planes(),
                "Invalid feature extractor used with dlib::scan_fhog_pyramid.  The output does not have the \n"
                "indicated number of planes.");


            //// 计算特征金字塔
            if (feats.size() > 1)
            {
                typedef typename image_traits<image_type>::pixel_type pixel_type;
                array2d<pixel_type> temp1, temp2;

                pyr(img, temp1);                                                         //创建图像金字塔
                fe(temp1, feats[1], cell_size, filter_rows_padding, filter_cols_padding);    //计算每一层图像金字塔的特征金字塔
                swap(temp1, temp2);

                for (unsigned long i = 2; i < feats.size(); ++i)
                {
                    //// 通过上一层的图像金字塔计算下一层的图像金字塔
                    pyr(temp2, temp1);
                    fe(temp1, feats[i], cell_size, filter_rows_padding, filter_cols_padding);
                    swap(temp1, temp2);
                }

            }

            //=======================DEBUG SHOW_PYR=======================//
            if (feats.size() > 4)
            {
                typedef typename image_traits<image_type>::pixel_type pixel_type;
                array2d<pixel_type> temp1, temp2,temp3,temp4;

                image_window win0(img, "original");
                image_window hogwin0(draw_fhog(feats[0]), "original hog");

                pyr(img, temp1);                                                    
                image_window win1(temp1, "pyr 1");
                image_window hogwin1(draw_fhog(feats[1]), "pyr1 hog");

                pyr(temp1, temp2);
                image_window win2(temp2, "pyr 2");
                image_window hogwin2(draw_fhog(feats[2]), "pyr2 hog");

                pyr(temp2, temp3);
                image_window win3(temp3, "pyr 3");
                image_window hogwin3(draw_fhog(feats[3]), "pyr3 hog");

                pyr(temp3, temp4);
                image_window win4(temp4, "pyr 4");
                image_window hogwin4(draw_fhog(feats[4]), "pyr4 hog");

                system("pause");
            }

        }


        //// detect_from_fhog_pyramid()函数
        //// 功能：1.对HOG特征金字塔的某一层滤波（果为31维滤波结果相加）
        ////       2.将该层满足阈值的滤波结果，从特征检测窗口，还原到原图
        ////       注意：一个滤波结果对应一个检测窗口
        //// 参数：- saliency_image   存放滤波后的结果
        ////       - area             没有进行0填充的矩形区域
        ////       - filter_padding   

        template <
            typename pyramid_type,
            typename feature_extractor_type,
            typename fhog_filterbank
            >
        void detect_from_fhog_pyramid (
            const array<array<array2d<float> > >& feats,
            const feature_extractor_type& fe,
            const fhog_filterbank& w,
            const double thresh,
            const unsigned long det_box_height,
            const unsigned long det_box_width,
            const int cell_size,
            const int filter_rows_padding,
            const int filter_cols_padding,
            std::vector<std::pair<double, rectangle> >& dets
        ) 
        {
            dets.clear();

            array2d<float> saliency_image;
            pyramid_type pyr;

            //// 遍历某一特征层
            // for all pyramid levels
            for (unsigned long l = 0; l < feats.size(); ++l)  
            {
                //// 每一特征层滤波后的结果
                const rectangle area = apply_filters_to_fhog(w, feats[l], saliency_image);  //非0填充的区域

                //// 遍历滤波结果，找出大于阈值的滤波结果，从特征的检测窗口还原到原图区域
                // now search the saliency image for any detections
                for (long r = area.top(); r <= area.bottom(); ++r)
                {
                    for (long c = area.left(); c <= area.right(); ++c)
                    {
                        // if we found a detection
                        if (saliency_image[r][c] >= thresh)
                        {
                            //// 将降采样后的特征层还原到原图
                            rectangle rect = fe.feats_to_image(centered_rect(point(c,r),det_box_width,det_box_height), 
                                cell_size, filter_rows_padding, filter_cols_padding);
                            rect = pyr.rect_up(rect, l); 
                            dets.push_back(std::make_pair(saliency_image[r][c], rect));
                        }
                    }
                }
            }

            //=======================DEBUG SHOW_FILTER_RES=======================//
            ////第l特征层滤波后的结果

            int level = 3;      //第几层特征层
            std::cout << " HOG Pyramid Level: " << level << endl;
            const rectangle area = apply_filters_to_fhog(w, feats[level], saliency_image);  //非0填充的区域
            for (long r = area.top(); r <= area.bottom(); ++r)
            {
                std::cout << " *  ";
                for (long c = area.left(); c <= area.right(); ++c)
                {
                    std::cout << fixed << setprecision(2) << saliency_image[r][c] << "  ";
                }
                std::cout << endl;
            }
            std::cout << endl << endl;

            std::sort(dets.rbegin(), dets.rend(), compare_pair_rect);   //从大到小排序
        }


        //// evaluate_detectors()函数
        //// 功能：目标检测

        template <
            typename pyramid_type,
            typename image_type
            >
        void evaluate_detectors (
            const std::vector<object_detector<scan_fhog_pyramid<pyramid_type> > >& detectors,
            const image_type& img,
            std::vector<rect_detection>& dets,
            const double adjust_threshold = 0
        )
        {
            typedef scan_fhog_pyramid<pyramid_type> scanner_type;

            dets.clear();
            if (detectors.size() == 0)
                return;

            const unsigned long cell_size = detectors[0].get_scanner().get_cell_size();

            // Find the maximum sized filters and also most extreme pyramiding settings used.
            unsigned long max_filter_width = 0;
            unsigned long max_filter_height = 0;
            unsigned long min_pyramid_layer_width = std::numeric_limits<unsigned long>::max();
            unsigned long min_pyramid_layer_height = std::numeric_limits<unsigned long>::max();
            unsigned long max_pyramid_levels = 0;
            bool all_cell_sizes_the_same = true;

            //// 检测窗口的尺寸：取所有detector最大
            //// 最小金字塔层的尺寸：取所有detector最小
            //// 判断每个scanner的cell size大小是否相同
            for (unsigned long i = 0; i < detectors.size(); ++i)
            {
                const scanner_type& scanner = detectors[i].get_scanner();
                max_filter_width = std::max(max_filter_width, scanner.get_fhog_window_width());      // 注意：这里的窗口是滤波器窗口不是像素窗口
                max_filter_height = std::max(max_filter_height, scanner.get_fhog_window_height());   //
                max_pyramid_levels = std::max(max_pyramid_levels, scanner.get_max_pyramid_levels());
                min_pyramid_layer_width = std::min(min_pyramid_layer_width, scanner.get_min_pyramid_layer_width());
                min_pyramid_layer_height = std::min(min_pyramid_layer_height, scanner.get_min_pyramid_layer_height());
                if (cell_size != scanner.get_cell_size())
                    all_cell_sizes_the_same = false;
            }

            //=======================DEBUG SHOW_FILTER =======================//
            //std::cout<<"max_filter_width "<<max_filter_width<<endl;   //注意：这里是针对特征的滤波器的尺寸，不再是像素级的尺寸
            //std::cout<<"max_filter_height "<<max_filter_height<<endl;
            //std::cout<<"all_cell_sizes_the_same "<<all_cell_sizes_the_same<<endl;
            //max_filter_height、weight没有什么实际作用，不论对hog特征金字塔用什么样的滤波器，滤波结果的尺寸，与hog的尺寸相同

            //// 所有detector的cell尺寸一致，只要计算一次pyramid HOG
            std::vector<rect_detection> dets_accum;     //检测框：检测框的位置、置信度、使用的是哪个detector
            // Do to the HOG feature extraction to make the fhog pyramid.  Again, note that we
            // are making a pyramid that will work with any of the detectors.  But only if all
            // the cell sizes are the same.  If they aren't then we have to calculate the
            // pyramid for each detector individually.
            
            array<array<array2d<float> > > feats;       //hog特征金字塔

            //// 若每个scanner的cell_size尺寸一样，计算一次特征金字塔
            if (all_cell_sizes_the_same)
            {
                impl::create_fhog_pyramid<pyramid_type>(img,
                    detectors[0].get_scanner().get_feature_extractor(), feats, cell_size,
                    max_filter_height, max_filter_width, min_pyramid_layer_width,
                    min_pyramid_layer_height, max_pyramid_levels);
            }

            std::vector<std::pair<double, rectangle> > temp_dets;   //置信度和检测到的目标框
            for (unsigned long i = 0; i < detectors.size(); ++i)    //对应第几个手势的detector[i]
            {
                //// 若scanner的cell size不同，每次重算特征金字塔
                const scanner_type& scanner = detectors[i].get_scanner();
                if (!all_cell_sizes_the_same)
                {
                    impl::create_fhog_pyramid<pyramid_type>(img,
                        scanner.get_feature_extractor(), feats, scanner.get_cell_size(),
                        max_filter_height, max_filter_width, min_pyramid_layer_width,
                        min_pyramid_layer_height, max_pyramid_levels);
                }
                
                //// 确定检测框的大小
                const unsigned long det_box_width  = scanner.get_fhog_window_width()  - 2*scanner.get_padding();    
                const unsigned long det_box_height = scanner.get_fhog_window_height() - 2*scanner.get_padding();

                //=======================DEBUG SHOW_BOX=======================//
                //std::cout << "detector " << i+1 << " : " << "box_width " << det_box_width << endl;
                //std::cout << "detector " << i+1 << " : " << "box_height " << det_box_height << endl << endl;

                //// 使用一个目标检测器，得到符合该检测器的目标框
                // A single detector object might itself have multiple weight vectors in it. So
                // we need to evaluate all of them.
                for (unsigned d = 0; d < detectors[i].num_detectors(); ++d)     //使用的第几个detector[i]中的第几个滤波器 //detectors[i].num_detectors()=1 即论文中使用的单个滤波器
                {
                    const double thresh = detectors[i].get_processed_w(d).w(scanner.get_num_dimensions());      //threshold 训练阈值


                    std::cout << " Gesture " << i+1 << " Detection " << endl;

                    impl::detect_from_fhog_pyramid<pyramid_type>(feats, scanner.get_feature_extractor(),
                        detectors[i].get_processed_w(d).get_detect_argument(), thresh+adjust_threshold,         //thresh + adjust_threshold，adjust_threshold=-0.3
                        det_box_height, det_box_width, cell_size, max_filter_height,
                        max_filter_width, temp_dets);

                    for (unsigned long j = 0; j < temp_dets.size(); ++j)
                    {
                        rect_detection temp;
                        temp.detection_confidence = temp_dets[j].first-thresh;
                        temp.weight_index = i;
                        temp.rect = temp_dets[j].second;
                        dets_accum.push_back(temp);
                    }
                }
            }

            //// 同一个detector得到的所有检测框，使用非极大值抑制合并
            // Do non-max suppression
            dets.clear();
            if (detectors.size() > 1)
                std::sort(dets_accum.rbegin(), dets_accum.rend());
            for (unsigned long i = 0; i < dets_accum.size(); ++i)
            {
                const test_box_overlap tester = detectors[dets_accum[i].weight_index].get_overlap_tester();     //注意：tester 是对应相应的detector[i]的
                                                                                                                //tester是object_detector类中的一个成员变量   boxes_overlap，该变量是test_box_overlap类 
                if (impl::overlaps_any_box(tester, dets, dets_accum[i]))
                    continue;

                dets.push_back(dets_accum[i]);
            }
        }



